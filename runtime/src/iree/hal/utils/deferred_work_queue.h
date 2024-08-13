// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_WORK_QUEUE_H_
#define IREE_HAL_UTILS_WORK_QUEUE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/semaphore_base.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_deferred_work_queue_t iree_hal_deferred_work_queue_t;

typedef struct iree_hal_deferred_work_queue_symbol_table_t
    iree_hal_deferred_work_queue_symbol_table_t;
typedef struct iree_hal_deferred_work_queue_symbol_table_vtable_t {
  void (*destroy)(iree_hal_deferred_work_queue_symbol_table_t*);
  iree_status_t (*bind_to_thread)(iree_hal_deferred_work_queue_symbol_table_t*);
  iree_status_t (*wait_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void*);
  iree_status_t (*create_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void**);
  iree_status_t (*record_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void*);
  iree_status_t (*synchronize_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void*);
  iree_status_t (*destroy_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void*);
  iree_status_t (*semaphore_acquire_timepoint_device_signal_native_event)(
      iree_hal_deferred_work_queue_symbol_table_t*,
      struct iree_hal_semaphore_t*, uint64_t, void**);
  bool (*acquire_host_wait_event)(iree_hal_deferred_work_queue_symbol_table_t*,
                                  struct iree_hal_semaphore_t*, uint64_t,
                                  void**);
  void (*release_wait_event)(iree_hal_deferred_work_queue_symbol_table_t*,
                             void*);
  void* (*native_event_from_wait_event)(
      iree_hal_deferred_work_queue_symbol_table_t*, void*);
  iree_status_t (*create_command_buffer_for_deferred)(
      iree_hal_deferred_work_queue_symbol_table_t*,
      iree_hal_command_buffer_mode_t, iree_hal_command_category_t,
      iree_hal_command_buffer_t**);
  iree_status_t (*submit_command_buffer)(
      iree_hal_deferred_work_queue_symbol_table_t*, iree_hal_command_buffer_t*);
} iree_hal_deferred_work_queue_symbol_table_vtable_t;

typedef struct iree_hal_deferred_work_queue_symbol_table_t {
  const iree_hal_deferred_work_queue_symbol_table_vtable_t* _vtable;
} iree_hal_deferred_work_queue_symbol_table_t;

iree_status_t iree_hal_deferred_work_queue_create(
    iree_hal_deferred_work_queue_symbol_table_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_deferred_work_queue_t** out_queue);

void iree_hal_deferred_work_queue_destroy(
    iree_hal_deferred_work_queue_t* queue);

typedef void(IREE_API_PTR* iree_hal_deferred_work_queue_cleanup_callback_t)(
    void* user_data);

iree_status_t iree_hal_deferred_work_queue_enque(
    iree_hal_deferred_work_queue_t* deferred_work_queue,
    iree_hal_deferred_work_queue_cleanup_callback_t cleanup_callback,
    void* callback_userdata,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables);

iree_status_t iree_hal_deferred_work_queue_issue(
    iree_hal_deferred_work_queue_t* deferred_work_queue);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  //  IREE_HAL_UTILS_WORK_QUEUE_H_
