// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include "iree/hal/utils/group_device.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"

#include "iree/hal/utils/group_command_buffer.h"

// group command buffer

typedef struct iree_hal_group_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  
  iree_allocator_t host_allocator;

  iree_hal_device_t** child_devices;
  iree_host_size_t child_device_count;

  iree_host_size_t queues_per_device;

  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;
  iree_hal_group_device_interface_t* interface;
  bool supports_memory_pools;
} iree_hal_group_device_t;

static const iree_hal_device_vtable_t iree_hal_group_device_vtable;


static iree_hal_group_device_t* iree_hal_group_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_group_device_vtable);
  return (iree_hal_group_device_t*)base_value;
}

static void iree_hal_group_device_destroy(iree_hal_device_t* base_device) {
    iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
    
    for (iree_host_size_t i = 0; i < device->child_device_count; ++i) {
        iree_hal_resource_release(device->child_devices[i]);
    }
    iree_allocator_free(device->host_allocator, device);
}

static iree_string_view_t iree_hal_group_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  return device->identifier;
}


static iree_allocator_t iree_hal_group_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_group_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  // TODO(awoloszyn), what goes here?
  return NULL;
}

static void iree_hal_group_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}


static void iree_hal_group_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}


static iree_status_t iree_hal_group_device_trim(iree_hal_device_t* base_device) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < device->child_device_count; ++i) {
    iree_status_join(status, iree_hal_device_trim(device->child_devices));
  }
  return status;
}


static iree_status_t iree_hal_group_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  // TODO(awoloszyn): Maybe we don't just want to return values for device[0]
  //                  should be fine as long as we only really care about homogenous 
  //                  devices.
  return iree_hal_device_query(device->child_devices[0], category, key, out_value);
}


static iree_status_t iree_hal_group_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED, 
        "TODO: implement the channel interface for group devices");
}

static iree_status_t iree_hal_group_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  
  // Never more than 64 child devices.
  iree_hal_command_buffer_t* child_cbs[64];
  iree_hal_command_buffer_t** child_cb;

  uint64_t queue_pos = 0;
  const uint64_t total_affinities = device->child_device_count * device->queues_per_device;
  const uint64_t all_affinities = 0xFFFFFFFFFFFFFFFF;

  iree_status_t status = iree_ok_status();
  while((queue_affinity >> queue_pos) > 0) {
    IREE_ASSERT_LT(queue_pos, total_affinities, "Queue afinity OOB");
    int ltz = iree_math_count_trailing_zeros_u64(queue_affinity);
    queue_pos += ltz;
    uint64_t device_num = queue_pos / device->queues_per_device;
    uint64_t affinities_for_this_device = queue_affinity >> (device_num * device->queues_per_device);
    affinities_for_this_device &= all_affinities >> (64 - device->queues_per_device);
    status = iree_status_join(status, iree_hal_command_buffer_create(device->child_devices[device_num], mode, command_categories, affinities_for_this_device, binding_capacity, child_cb));
    child_cb++;
  }
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    return status;
  }

  return iree_hal_utils_group_command_buffer_create(device->host_allocator, 
    child_cb - &child_cbs[0], child_cbs);
}

static iree_status_t iree_hal_group_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implmeneted");
}

static iree_status_t iree_hal_group_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_group_device_t* device = iree_hal_group_device_cast(base_device);
  (*interface->vtable->create_executable_cache)(interface,
    identifier, loop, out_executable_cache
  );
}

static const iree_hal_device_vtable_t iree_hal_group_device_vtable = {
    .destroy = iree_hal_group_device_destroy,
    .id = iree_hal_group_device_id,
    .host_allocator = iree_hal_group_device_host_allocator,
    .device_allocator = iree_hal_group_device_allocator,
    .replace_device_allocator = iree_hal_group_replace_device_allocator,
    .replace_channel_provider = iree_hal_group_replace_channel_provider,
    .trim = iree_hal_group_device_trim,
    .query_i64 = iree_hal_group_device_query_i64,
    .create_channel = iree_hal_group_device_create_channel,
    .create_command_buffer = iree_hal_group_device_create_command_buffer,
    .create_event = iree_hal_group_device_create_event,
    .create_executable_cache = iree_hal_group_device_create_executable_cache,
    .import_file = iree_hal_group_device_import_file,
    .create_semaphore = iree_hal_group_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_group_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_group_device_queue_alloca,
    .queue_dealloca = iree_hal_group_device_queue_dealloca,
    .queue_read = iree_hal_group_device_queue_read,
    .queue_write = iree_hal_group_device_queue_write,
    .queue_execute = iree_hal_group_device_queue_execute,
    .queue_flush = iree_hal_group_device_queue_flush,
    .wait_semaphores = iree_hal_group_device_wait_semaphores,
    .profiling_begin = iree_hal_group_device_profiling_begin,
    .profiling_flush = iree_hal_group_device_profiling_flush,
    .profiling_end = iree_hal_group_device_profiling_end,
};
