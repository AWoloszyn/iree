#include "iree/hal/drivers/hip/cleanup_thread.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/queue.h"

static const iree_host_size_t iree_hal_hip_cleanup_thread_default_queue_size =
    64;

typedef struct iree_hal_hip_cleanup_thread_callback_t {
  iree_hal_hip_cleanup_callback_t callback;
  void* user_data;
  iree_hal_hip_event_t* event;
};

IREE_TYPED_QUEUE_WRAPPER(iree_hal_hip_callback_queue,
                         iree_hal_hip_cleanup_thread_callback_t,
                         iree_hal_hip_cleanup_thread_default_queue_size);

typedef struct iree_hal_hip_cleanup_thread_t {
  iree_thread_t* thread;
  const iree_hal_hip_dynamic_symbols_t* symbols;
  iree_slim_mutex_t mutex;

  iree_hal_hip_callback_queue_t queue;
  iree_status_t failure_status;
  iree_notification_t notification;
} iree_hal_hip_cleanup_thread_t;

static int iree_hal_hip_cleanup_thread_main(void* param) {
  iree_hal_hip_cleanup_thread_t* thread = (iree_hal_hip_cleanup_thread_t*)param;
  while (true) {
    iree_notification_await(&thread->notification, IREE_TIME_INFINITE_FUTURE);

    iree_slim_mutex_lock(&thread->mutex);
    while (!iree_hal_hip_callback_queue_empty(&thread->queue)) {
      iree_hal_hip_cleanup_thread_callback_t callback;
      iree_hal_hip_callback_queue_pop_front(&thread->queue, &callback);
      iree_status_t status = thread->failure_status;
      iree_slim_mutex_unlock(&thread->mutex);

      if (iree_status_is_ok(status)) {
        status = IREE_HIP_RESULT_TO_STATUS(
            thread->symbols,
            hipEventSynchronize(iree_hal_hip_event_handle(callback.event)));
      }
      iree_hal_hip_event_release(callback.event);

      status = iree_status_join(
          callback.callback(callback.user_data, callback.event, status));
      iree_slim_mutex_lock(&thread->mutex);
      if (!iree_status_is_ok(status)) {
        thread->failure_status = status;
      }
      iree_slim_mutex_lock(&thread->mutex);
    }
    iree_slim_mutex_unlock(&thread->mutex);

    if (!iree_status_is_ok(thread->failure_status)) {
      break;
    }
  }
  return 0;
}

iree_status_t iree_hal_hip_cleanup_thread_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_cleanup_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_cleanup_thread_t* thread;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*thread), (void**)&thread));

  thread->symbols = symbols;
  iree_slim_mutex_initialize(&thread->mutex);
  iree_hal_hip_callback_queue_initialize(&thread->queue, host_allocator);
  thread->failure_status = iree_ok_status();
  iree_status_t status =
      iree_thread_create(iree_make_cstring_view("iree-hal-hip-cleanup"),
                         (iree_thread_entry_t)iree_hal_hip_cleanup_thread_main,
                         thread, host_allocator, &thread->thread);
  if (!iree_status_is_ok(status)) {
    iree_hal_hip_callback_queue_deinitialize(&thread->queue);
    iree_slim_mutex_deinitialize(&thread->mutex);
    iree_allocator_free(thread->host_allocator, thread);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Deinitializes the cleanup thread for HIP driver.
void iree_hal_hip_cleanup_thread_deinitialize(
    iree_hal_hip_cleanup_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // There is only one owner for the thread, so this also joins the thread.
  iree_thread_release(thread->thread);

  iree_hal_hip_callback_queue_deinitialize(&thread->queue);
  iree_slim_mutex_deinitialize(&thread->mutex);
  iree_allocator_free(thread->host_allocator, thread);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t add_cleanup(iree_hal_hip_cleanup_thread_t* thread,
                          iree_hal_hip_event_t* event,
                          iree_hal_hip_cleanup_callback_t callback,
                          void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&thread->mutex);
  if (!iree_status_is_ok(thread->failure_status)) {
    IREE_TRACE_ZONE_END(z0);
    iree_slim_mutex_unlock(&thread->mutex);
    return thread->failure_status;
  }

  iree_hal_hip_cleanup_thread_callback_t callback = {
      .callback = callback, .user_data = user_data, .event = event};
  iree_hal_hip_callback_queue_push_back(&thread->queue, callback);
  iree_slim_mutex_unlock(&thread->mutex);
  iree_notification_post(&thread->notification, IREE_ALL_WAITERS);

  IREE_TRACE_ZONE_END(z0);
}
