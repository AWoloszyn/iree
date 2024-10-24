// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/event_semaphore.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/rbt.h"
#include "iree/base/status.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

typedef struct iree_hal_hip_cpu_event_t {
  iree_hal_resource_t resource;
  iree_event_t event;
} iree_hal_hip_cpu_event_t;

typedef struct iree_hal_hip_semaphore_work_item_t {
  iree_hal_hip_event_semaphore_scheduled_callback_t scheduled_callback;
  void* user_data;
  iree_hal_hip_semaphore_work_item_t* next;
} iree_hal_hip_semaphore_work_item_t;

typedef struct iree_hal_hip_semaphore_queue_item_t {
  iree_hal_hip_event_t* event;
  iree_hal_hip_cpu_event_t* cpu_event;
  iree_hal_hip_semaphore_work_item_t* work_item;
} iree_hal_hip_semaphore_queue_item_t;

typedef struct iree_hal_hip_semaphore_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t base;

  // The allocator used to create this semaphore.
  iree_allocator_t host_allocator;
  // The symbols used to issue HIP API calls.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // The queue of hip events that back any GPU signals of this semaphore.
  iree_tree_t event_queue;
  uint8_t event_queue_precache[1024];

  // Notify any potential CPU waiters that this semaphore
  // has changed state.
  iree_notification_t state_notification;

  iree_slim_mutex_t mutex;
  // The maximum value that this semaphore has been signaled to.
  // This means this semaphore is guaranteed to make forward progress
  // until that semaphore is hit, as all signaling operations have
  // been made available.
  uint64_t max_value_to_be_signaled IREE_GUARDED_BY(mutex);

  // The largest value that has been observed by the host.
  uint64_t current_visible_value IREE_GUARDED_BY(mutex);

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status IREE_GUARDED_BY(mutex);
} iree_hal_hip_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_hip_semaphore_vtable;

static iree_hal_hip_semaphore_t* iree_hal_hip_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_semaphore_vtable);
  return (iree_hal_hip_semaphore_t*)base_value;
}

static iree_status_t iree_hal_hip_event_semaphore_advance(
    iree_hal_hip_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Take out all of the values from the queue that are less than the
  // current visible value, and make sure we advance any work needed
  // on them.
  status = iree_ok_status();
  bool updated = false;
  do {
    iree_mutex_lock(&semaphore->mutex);
    iree_tree_node_t* node = iree_tree_first(&semaphore->event_queue);
    if (node == NULL) {
      break;
    }
    if (iree_tree_node_get_key(node) > semaphore->current_visible_value) {
      iree_mutex_unlock(&semaphore->mutex);
      break;
    }
    iree_hal_hip_semaphore_queue_item_t copy =
        *(iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node);
    iree_tree_erase(&semaphore->event_queue, node);
    iree_mutex_unlock(&semaphore->mutex);

    iree_hal_hip_event_release(copy->event);
    if (copy->cpu_event) {
      iree_event_set(&copy->cpu_event->event);
      iree_hal_resource_release(&copy->cpu_event->resource);
    }
    updated = true;
  } while (true);

  if (updated) {
    iree_notification_post(&semaphore->state_notification, IREE_ALL_WAITERS);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_semaphore_notify_forward_progress_to(
    iree_hal_semaphore_t* base_semaphore, uint64_t value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_mutex_lock(&semaphore->mutex);
  iree_status_t status = semaphore->failure_status;
  if (!iree_status_is_ok(status)) {
    iree_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_hip_semaphore_work_item_t* work_item = NULL;
  iree_hal_hip_semaphore_work_item_t* last_work_item = NULL;
  if (value > semaphore->max_value_to_be_signaled) {
    iree_tree_node_t* node = iree_tree_upper_bound(
        &semaphore->event_queue, semaphore->max_value_to_be_signaled);
    // Collect all of the things to schedule now that we know we can safely make
    // it to a given value.
    while (node && iree_tree_node_get_key(node) <= value) {
      iree_hal_hip_semaphore_queue_item_t* queue_item = ((
          iree_hal_hip_semaphore_queue_item_t*))iree_tree_node_get_value(node);
      iree_hal_hip_semaphore_work_item_t* next_work_item =
          queue_item->work_item;
      while (next_work_item) {
        if (!work_item) {
          work_item = next_work_item;
        }
        if (!last_work_item->next) {
          last_work_item->next = next_work_item;
        }
        last_work_item = next_work_item;
        next_work_item = next_work_item->next;
      }
      queue_item->work_item = NULL;
    }
  }

  iree_mutex_unlock(&semaphore->mutex);
  IREE_TRACE_ZONE_END(z0);

  // Now that we have accumulated all of the work items, and we have
  // unlocked the semaphore, start running through the work items.
  while (work_item) {
    iree_hal_hip_semaphore_work_item_t* next_work_item = work_item->next;
    work_item->scheduled_callback(work_item->user_data, base_semaphore, status);
    iree_allocator_free(semaphore->host_allocator, work_item);
    work_item = next_work_item;
  }
  return status;
}

iree_status_t iree_hal_hip_event_semaphore_create(
    uint64_t initial_value, const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                (void**)&semaphore));

  iree_hal_semaphore_initialize(&iree_hal_hip_semaphore_vtable,
                                &semaphore->base);
  semaphore->host_allocator = host_allocator;
  semaphore->symbols = symbols;
  iree_tree_initialize(&semaphore->event_queue,
                       sizeof(iree_hal_hip_semaphore_queue_item_t),
                       semaphore->event_queue_precache,
                       sizeof(semaphore->event_queue_precache), host_allocator);
  iree_notification_initialize(&semaphore->state_notification);

  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->current_visible_value = initial_value;
  semaphore->max_value_to_be_signaled = initial_value;
  semaphore->failure_status = iree_ok_status();

  *out_semaphore = &semaphore->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(semaphore->failure_status);
  iree_slim_mutex_deinitialize(&semaphore->mutex);

  iree_notification_deinitialize(&working_area->state_notification);
  for (iree_tree_node_t* i = iree_tree_first(&semaphore->event_queue);
       t != NULL; i = iree_tree_node_next(i)) {
    iree_hal_hip_event_release(
        ((iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(i))
            ->event);
  }
  iree_tree_deinitialize(&semaphore->event_queue);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hip_semaphore_query_locked(
    iree_hal_hip_semaphore_t* semaphore, uint64_t* out_value) {
  *out_value = semaphore->current_visible_value;
  uint64_t i = 0;
  iree_tree_node_t* node = iree_tree_first(&semaphore->event_queue);
  while (node) {
    iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
        semaphore->symbols,
        hipEventQuery(iree_hal_hip_event_handle(
            ((iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(
                 node))
                ->event)));
    if (!iree_status_is_ok(status)) {
      break;
    }
    *out_value += iree_tree_node_get_key(node);
    iree_hal_hip_event_release(iree_hal_hip_event_handle(
        ((iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node))
            ->event));
    iree_tree_erase(&semaphore->event_queue, node);
    node = iree_tree_first(&semaphore->event_queue);
  }

  if (semaphore->current_visible_value < *out_value) {
    semaphore->current_visible_value = *out_value;
    iree_notification_post(&semaphore->state_notification, IREE_ALL_WAITERS);
  }

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->mutex);

  ireee_status_t status =
      iree_hal_hip_semaphore_query_locked(semaphore, out_value);

  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_hip_event_semaphore_advance(semaphore);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->mutex);

  if (new_value <= semaphore->current_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore values must be monotonically "
                            "increasing; current_value=%" PRIu64
                            ", new_value=%" PRIu64,
                            current_value, new_value);
  }

  semaphore->current_value = new_value;

  iree_slim_mutex_unlock(&semaphore->mutex);

  iree_hal_hip_event_semaphore_advance(semaphore);
  iree_hal_hip_semaphore_notify_forward_progress_to(semaphore, new_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hip_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                        iree_status_t status) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_status_code_t status_code = iree_status_code(status);

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous sta-tus was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_visible_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_hip_event_semaphore_advance(semaphore);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hip_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_slim_mutex_lock(&semaphore->mutex);
  uint64_t ignored_current_value;

  // query_locked to make sure our count is up to date.
  iree_status_t status =
      iree_hal_hip_semaphore_query_locked(semaphore, &ignored_current_value);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  while (semaphore->max_value_to_be_signaled < value) {
    iree_wait_token_t wait =
        iree_notification_prepare_wait(&semaphore->state_notification);
    iree_slim_mutex_unlock(&semaphore->mutex);

    // We have to wait for the semaphore to catch up.
    if (!iree_notification_commit_wait(&semaphore->state_notification, wait,
                                       IREE_DURATION_ZERO, deadline_ns)) {
      return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED);
    }

    iree_slim_mutex_unlock(&semaphore->mutex);

    // query_locked to make sure our count is up to date.
    status =
        iree_hal_hip_semaphore_query_locked(semaphore, &ignored_current_value);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // The current value stored in the semaphore is greater than the current
  // value, so we can return.
  if (semaphore->current_visible_value >= value) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // The current value is not enough, but we have at least submitted
  // the work that will increment the semaphore to the value we need.
  // Use iree_tree_lower_bound to find the first element in the tree that would
  // signal our semaphore to at least the given value.
  iree_tree_node_t* node =
      iree_tree_lower_bound(&semaphore->event_queue, value);
  IREE_ASSERT(
      node,
      "We really should either have an event in the queue that will satisfy"
      "this semaphore, (we checked max_value_to_be_signaled above), or we"
      "should already have signaled (current_visible_value above)");
  iree_hal_hip_semaphore_queue_item_t* item =
      (iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node);
  // TODO(awoloszyn): This turns the rest of this into an infinite wait
  // and ignores the timeout because hipEvent_t does not support.
  // However we have a straight-forward fix to this (to be done in a
  // follow-up), which is, in the case of a non-infinite wait, re-use the loop
  // above to wait until the current_visible_value >= value, instead of the
  // max_value_to_be_signaled.
  iree_hal_hip_event_t* event = iree_hal_hip_event_handle(item->event);

  // Retain the event, as the event may be removed from the tree
  // while we sleep on the event.
  iree_hal_hip_event_retain(event);
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      semaphore->symbols,
      hipEventSynchronize(iree_hal_hip_event_handle(event)));
  iree_hal_hip_event_release(event);
  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->current_visible_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_semaphore_multi_wait(
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_wait_mode_t wait_mode, iree_timeout_t timeout,
    iree_arena_block_pool_t* block_pool) {
  if (semaphore_list.count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status = iree_ok_status();

  // TODO in a follow-up fix IREE_HAL_WAIT_MODE_ANY, and turn this into
  // just a fast-path. We have the events ready already.
  // if (semaphore_list.count == 1 || wait_mode == IREE_HAL_WAIT_MODE_ALL) {
  //  // Fast-path if we don't have to wait on only a subset of the semaphores.
  iree_timeout_t t = iree_make_deadline(deadline_ns);
  status = iree_status_join(iree_hal_semaphore_wait(
      semaphore_list.semaphores[0], semaphore_list.payload_values[0], t));
  if (!iree_status_is_ok(status)) {
    return status;
  }
  //}

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_semaphore_notify_work(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_hip_event_semaphore_scheduled_callback_t callback,
    void* user_data) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_mutex_lock(&semaphore->mutex);
  iree_status_t status = semaphore->failure_status();
  if (!iree_status_is_ok(status)) {
    iree_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (value > semaphore->max_value_to_be_signaled) {
    iree_tree_node_t* node = iree_tree_get(&semaphore->event_queue, value);
    if (node == NULL) {
      iree_status_t status = iree_tree_insert(tree, value, &node);
      if (!iree_status_is_ok(status)) {
        iree_mutex_unlock(&semaphore->mutex);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      iree_hal_hip_semaphore_queue_item_t* item =
          (iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node);
      item->event = iree_hal_hip_event_get();
      item->cpu_event = NULL;
      item->work_item = NULL;
    }
    iree_hal_hip_semaphore_queue_item_t* item =
        (iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node);
    iree_hal_hip_semaphore_work_item_t* work_item;
    iree_status_t status = iree_allocator_malloc(
        semaphore->host_allocator, sizeof(*work_item), (void**)&work_item);
    if (!iree_status_is_ok(status)) {
      iree_mutex_unlock(&semaphore->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    work_item->scheduled_callback = callback;
    work_item->user_data = user_data;
    work_item->next = item->work_item;
    item->work_item = work_item;
    callback = NULL;
  }
  iree_mutex_unlock(&semaphore->mutex);

  // If this semaphore requirement has already been satisfied, then
  // we can just run the callback right now.
  if (callback) {
    status = callback(user_data, base_semaphore, status);
  }
  return status;
}

iree_status_t iree_hal_hip_semaphore_get_hip_event(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    hipEvent_t* out_hip_event) {
  iree_hal_hip_semaphore_t* semaphore =
      iree_hal_hip_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_mutex_lock(&semaphore->mutex);
  if (value < semaphore->current_visible_value) {
    iree_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    out_hip_event[0] = NULL;
    return iree_ok_status();
  }
  iree_status_t status = semaphore->failure_status;
  if (!iree_status_is_ok(status)) {
    iree_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_tree_node_t* node = iree_tree_get(&semaphore->event_queue, value);
  if (node == NULL) {
    iree_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "The given value has not been notified as of yet");
  }
  iree_hal_hip_event_t* event =
      ((iree_hal_hip_semaphore_queue_item_t*)iree_tree_node_get_data(node))
          ->event;
  iree_hal_hip_event_retain(event);
  *out_hip_event = event;
  return iree_ok_status();
}

static const iree_hal_semaphore_vtable_t iree_hal_hip_semaphore_vtable = {
    .destroy = iree_hal_hip_semaphore_destroy,
    .query = iree_hal_hip_semaphore_query,
    .signal = iree_hal_hip_semaphore_signal,
    .fail = iree_hal_hip_semaphore_fail,
    .wait = iree_hal_hip_semaphore_wait,
};
