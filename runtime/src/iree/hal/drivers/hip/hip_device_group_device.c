// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_device_group_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/event_semaphore.h"
#include "iree/hal/drivers/hip/graph_command_buffer.h"
#include "iree/hal/drivers/hip/hip_allocator.h"
#include "iree/hal/drivers/hip/hip_device_group_command_buffer.h"
#include "iree/hal/drivers/hip/memory_pools.h"
#include "iree/hal/drivers/hip/nop_executable_cache.h"
#include "iree/hal/drivers/hip/per_device_information.h"
#include "iree/hal/drivers/hip/rccl_channel.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/stream_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/stream_tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_hip_device_group_device_t
//===----------------------------------------------------------------------===//

typedef enum iree_hip_device_group_device_commandbuffer_type_e {
  IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_STREAM,
  IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH,
} iree_hip_device_group_device_commandbuffer_type_e;

typedef struct iree_hal_hip_device_group_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the HIP symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols;

  // Parameters used to control device behavior.
  iree_hal_hip_device_params_t params;

  iree_allocator_t host_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;

  // Timepoint pools, shared by various semaphores.
  iree_hal_hip_timepoint_pool_t* timepoint_pool;

  // Device memory pools and allocators.
  bool supports_memory_pools;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  iree_hal_allocator_t* device_allocator;

  iree_hal_hip_memory_pools_t memory_pools;

  // The number of underlying devices in this device_group.
  uint32_t num_physical_devices;
  iree_hal_hip_per_device_information_t device_contexts[];
} iree_hal_hip_device_group_device_t;

static const iree_hal_device_vtable_t iree_hal_hip_device_group_device_vtable;

static iree_status_t
iree_hal_hip_device_group_device_create_command_buffer_internal(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hip_device_group_device_commandbuffer_type_e type,
    iree_hal_command_buffer_t** out_command_buffer);

typedef struct iree_hal_hip_tracing_device_group_device_interface_t {
  iree_hal_stream_tracing_device_interface_t base;
  iree_hal_hip_per_device_information_t* device_context;
  iree_allocator_t host_allocator;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
} iree_hal_hip_tracing_device_group_device_interface_t;
static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_group_device_interface_vtable_t;

void iree_hal_hip_tracing_device_group_device_interface_destroy(
    iree_hal_stream_tracing_device_interface_t* base_device_interface) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  iree_allocator_free(device_interface->host_allocator, device_interface);
}

iree_status_t
iree_hal_hip_tracing_device_group_device_interface_synchronize_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventSynchronize((hipEvent_t)base_event));
}

iree_status_t
iree_hal_hip_tracing_device_group_device_interface_create_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t* base_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipEventCreateWithFlags((hipEvent_t*)base_event, hipEventDefault));
}

iree_status_t
iree_hal_hip_tracing_device_group_device_interface_query_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventQuery((hipEvent_t)base_event));
}

void iree_hal_hip_tracing_device_group_device_interface_event_elapsed_time(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    float* relative_millis, iree_hal_stream_tracing_native_event_t start_event,
    iree_hal_stream_tracing_native_event_t end_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  IREE_HIP_IGNORE_ERROR(
      device_interface->hip_symbols,
      hipEventElapsedTime(relative_millis, (hipEvent_t)start_event,
                          (hipEvent_t)end_event));
}

void iree_hal_hip_tracing_device_group_device_interface_destroy_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  IREE_HIP_IGNORE_ERROR(device_interface->hip_symbols,
                        hipEventDestroy((hipEvent_t)base_event));
}

iree_status_t
iree_hal_hip_tracing_device_group_device_interface_record_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipEventRecord(
          (hipEvent_t)base_event,
          (hipStream_t)device_interface->device_context->hip_dispatch_stream));
}

iree_status_t
iree_hal_hip_tracing_device_group_device_interface_add_graph_event_record_node(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_graph_node_t* out_node,
    iree_hal_stream_tracing_native_graph_t graph,
    iree_hal_stream_tracing_native_graph_node_t* dependency_nodes,
    size_t dependency_nodes_count,
    iree_hal_stream_tracing_native_event_t event) {
  iree_hal_hip_tracing_device_group_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_group_device_interface_t*)
          base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipGraphAddEventRecordNode((hipGraphNode_t*)out_node, (hipGraph_t)graph,
                                 (hipGraphNode_t*)dependency_nodes,
                                 dependency_nodes_count, (hipEvent_t)event));
}

static iree_hal_hip_device_group_device_t*
iree_hal_hip_device_group_device_cast(iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_device_group_device_vtable);
  return (iree_hal_hip_device_group_device_t*)base_value;
}

static iree_hal_hip_device_group_device_t*
iree_hal_hip_device_group_device_cast_unsafe(iree_hal_device_t* base_value) {
  return (iree_hal_hip_device_group_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hip_device_params_initialize(
    iree_hal_hip_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  out_params->allow_inline_execution = false;
}

static iree_status_t iree_hal_hip_device_group_device_check_params(
    const iree_hal_hip_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_group_device_initialize_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params,
    iree_hal_hip_device_group_device_t* device,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_allocator_t host_allocator) {
  const iree_host_size_t identifier_offset =
      sizeof(*device) + sizeof(iree_hal_hip_per_device_information_t) *
                            device->num_physical_devices;

  iree_hal_resource_initialize(&iree_hal_hip_device_group_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hip_symbols = symbols;
  device->nccl_symbols = nccl_symbols;
  device->params = *params;

  device->host_allocator = host_allocator;
  iree_status_t status;
  // Enable tracing for each of the streams - no-op if disabled.
  if (iree_status_is_ok(status) && device->params.stream_tracing) {
    if (device->params.stream_tracing >=
            IREE_HAL_STREAM_TRACING_VERBOSITY_MAX ||
        device->params.stream_tracing < IREE_HAL_STREAM_TRACING_VERBOSITY_OFF) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid stream_tracing argument: expected to be between %d and %d",
          IREE_HAL_STREAM_TRACING_VERBOSITY_OFF,
          IREE_HAL_STREAM_TRACING_VERBOSITY_MAX);
    }

    for (uint32_t i = 0; i < device->num_physical_devices; ++i) {
      iree_hal_hip_tracing_device_group_device_interface_t*
          tracing_device_interface;
      status = iree_allocator_malloc(
          host_allocator,
          sizeof(iree_hal_hip_tracing_device_group_device_interface_t),
          (void**)&tracing_device_interface);

      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        iree_hal_device_release((iree_hal_device_t*)device);
        return status;
      }

      tracing_device_interface->base.vtable =
          &iree_hal_hip_tracing_device_group_device_interface_vtable_t;
      tracing_device_interface->device_context = &device->device_contexts[i];
      tracing_device_interface->host_allocator = host_allocator;
      tracing_device_interface->hip_symbols = symbols;

      status = IREE_HIP_RESULT_TO_STATUS(
          symbols, hipCtxPushCurrent(device->device_contexts[i].hip_context));
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        iree_hal_device_release((iree_hal_device_t*)device);
        return status;
      }
      status = iree_hal_stream_tracing_context_allocate(
          (iree_hal_stream_tracing_device_interface_t*)tracing_device_interface,
          device->identifier, device->params.stream_tracing,
          &device->block_pool, host_allocator,
          &device->device_contexts[i].tracing_context);
      status = IREE_HIP_RESULT_TO_STATUS(symbols, hipCtxPopCurrent(NULL));
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        iree_hal_device_release((iree_hal_device_t*)device);
        return status;
      }
    }
  }

  // Memory pool support is conditional.
  if (iree_status_is_ok(status) && params->async_allocations) {
    device->supports_memory_pools = true;
    for (uint32_t i = 0; i < device->num_physical_devices; ++i) {
      int supports_memory_pools = 0;
      status = IREE_HIP_RESULT_TO_STATUS(
          symbols,
          hipDeviceGetAttribute(&supports_memory_pools,
                                hipDeviceAttributeMemoryPoolsSupported,
                                device->device_contexts[i].hip_device),
          "hipDeviceGetAttribute");
      device->supports_memory_pools &= (supports_memory_pools != 0);
    }
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status) && device->supports_memory_pools) {
    device->supports_memory_pools = false;
    // TODO (awoloszyn): Figure out how to set up memory pools in a device group
    // status = iree_hal_hip_memory_pools_initialize(
    //     symbols, hip_devices[i], &params->memory_pools, host_allocator,
    //     &device->device_contexts[i].memory_pools);
  }

  status = iree_hal_hip_allocator_create(
      symbols, device->num_physical_devices, device->device_contexts,
      device->supports_memory_pools ? &device->memory_pools : NULL,
      host_allocator, &device->device_allocator);

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_hip_device_group_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    uint32_t device_count, hipDevice_t* devices,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_device_group_device_t* device;
  const iree_host_size_t total_device_size =
      sizeof(*device) +
      sizeof(iree_hal_hip_per_device_information_t) * device_count +
      identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_device_size,
                                             (void**)&device));
  device->num_physical_devices = device_count;

  iree_status_t status = iree_hal_hip_device_group_device_check_params(params);

  // Get the main context for the device.
  for (uint32_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    device->device_contexts[i].hip_device = devices[i];
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols, hipDevicePrimaryCtxRetain(
                     &device->device_contexts[i].hip_context, devices[i]));
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_RESULT_TO_STATUS(
          symbols, hipCtxSetCurrent(device->device_contexts[i].hip_context));
    }

    // Create the default dispatch stream for the device.
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_RESULT_TO_STATUS(
          symbols, hipStreamCreateWithFlags(
                       &device->device_contexts[i].hip_dispatch_stream,
                       hipStreamNonBlocking));
    }

    if (iree_status_is_ok(status)) {
      for (uint32_t j = 0; j < device_count && iree_status_is_ok(status); ++j) {
        if (i == j) {
          continue;
        }
        status = IREE_HIP_RESULT_TO_STATUS(
            symbols, hipDeviceEnablePeerAccess(devices[j], 0));
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_group_device_initialize_internal(
        driver, identifier, params, device, symbols, nccl_symbols,
        host_allocator);
  } else {
    for (uint32_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
      if (device->device_contexts[i].hip_dispatch_stream)
        symbols->hipStreamDestroy(
            device->device_contexts[i].hip_dispatch_stream);
      // NOTE: This function return hipSuccess though doesn't release the
      // primaryCtx by design on HIP/HCC path.
      if (device->device_contexts[i].hip_context)
        symbols->hipDevicePrimaryCtxRelease(
            device->device_contexts[i].hip_device);
    }
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  for (uint32_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    if (iree_status_is_ok(status)) {
      status = iree_hal_hip_event_pool_allocate(
          symbols, params->event_pool_capacity, host_allocator,
          device->device_contexts[i].hip_context,
          &device->device_contexts[i].device_event_pool);
    }
  }

  iree_hal_hip_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_timepoint_pool_allocate(
        host_event_pool, device_count, device->device_contexts,
        params->event_pool_capacity, host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    device->host_event_pool = host_event_pool;
    device->timepoint_pool = timepoint_pool;
  } else {
    // Release resources we have accquired after HAL device creation.
    if (timepoint_pool) iree_hal_hip_timepoint_pool_free(timepoint_pool);
    for (uint32_t i = 0; i < device_count; ++i) {
      if (device->device_contexts[i].device_event_pool)
        iree_hal_hip_event_pool_release(
            device->device_contexts[i].device_event_pool);
    }
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    // Release other resources via the HAL device.
    iree_hal_device_release((iree_hal_device_t*)device);
    device = NULL;
  }
  *out_device = (iree_hal_device_t*)device;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_hip_dynamic_symbols_t*
iree_hal_hip_device_group_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast_unsafe(base_device);
  return device->hip_symbols;
}

static void iree_hal_hip_device_group_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  const iree_hal_hip_dynamic_symbols_t* symbols = device->hip_symbols;
  IREE_TRACE_ZONE_BEGIN(z0)

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_hip_memory_pools_deinitialize(&device->memory_pools);
  for (uint32_t i = 0; i < device->num_physical_devices; ++i) {
    iree_hal_stream_tracing_context_free(
        device->device_contexts[i].tracing_context);
  }

  // Destroy various pools for synchronization.
  if (device->timepoint_pool) {
    iree_hal_hip_timepoint_pool_free(device->timepoint_pool);
  }
  for (uint32_t i = 0; i < device->num_physical_devices; ++i) {
    if (device->device_contexts[i].device_event_pool) {
      iree_hal_hip_event_pool_release(
          device->device_contexts[i].device_event_pool);
    }
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  for (uint32_t i = 0; i < device->num_physical_devices; ++i) {
    IREE_HIP_IGNORE_ERROR(
        symbols,
        hipStreamDestroy(device->device_contexts[i].hip_dispatch_stream));
    // NOTE: This function return hipSuccess though doesn't release the
    // primaryCtx by design on HIP/HCC path.
    IREE_HIP_IGNORE_ERROR(symbols, hipDevicePrimaryCtxRelease(
                                       device->device_contexts[i].hip_device));
  }

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hip_device_group_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hip_device_group_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hip_device_group_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hip_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hip_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_hip_device_group_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  if (device->supports_memory_pools) {
    IREE_RETURN_IF_ERROR(iree_hal_hip_memory_pools_trim(
        &device->memory_pools, &device->params.memory_pools));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_group_device_query_attribute(
    iree_hal_hip_device_group_device_t* device, hipDeviceAttribute_t attribute,
    int64_t* out_value) {
  int value = 0;
  IREE_HIP_RETURN_IF_ERROR(
      device->hip_symbols,
      hipDeviceGetAttribute(&value, attribute,
                            device->device_contexts[0].hip_device),
      "hipDeviceGetAttribute");
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_group_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("rocm-hsaco-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hip_device_group_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  if (!device->nccl_symbols || !device->nccl_symbols->dylib) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "RCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (rccl.dll/librccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  // Today we only allow a single logical device per channel.
  // We could multiplex channels but it'd be better to surface that to the
  // compiler so that it can emit the right rank math.
  int requested_count = iree_math_count_ones_u64(queue_affinity);
  // TODO(#12206): properly assign affinity in the compiler.
  if (requested_count != 64 && requested_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exactly one participant is allowed in a "
                            "channel but %d were specified",
                            requested_count);
  }

  // Ask the channel provider (if configured) for the default rank and count
  // if the user did not set them.
  if (device->channel_provider &&
      (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT ||
       params.count == IREE_HAL_CHANNEL_COUNT_DEFAULT)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_channel_provider_query_default_rank_and_count(
            device->channel_provider, &params.rank, &params.count),
        "querying default collective group rank and count");
  }

  // An ID is required to initialize NCCL. On the root it'll be the local ID and
  // on all other participants it'll be the root ID.
  iree_hal_hip_nccl_id_t id;
  memset(&id, 0, sizeof(id));
  if (iree_const_byte_span_is_empty(params.id)) {
    // User wants the default ID.
    if (!device->channel_provider) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "default collective channel ID requested but no channel provider has "
          "been set on the device to provide it");
    }
    if (params.rank == 0) {
      // Bootstrap NCCL to get the root ID.
      IREE_RETURN_IF_ERROR(
          iree_hal_hip_nccl_get_unique_id(device->nccl_symbols, &id),
          "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_IF_ERROR(iree_hal_channel_provider_exchange_default_id(
                             device->channel_provider,
                             iree_make_byte_span((void*)&id, sizeof(id))),
                         "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    // User provided something but it's not what we expect.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NCCL ID must be %zu bytes matching the "
                            "ncclUniqueId struct but caller provided %zu bytes",
                            IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_hip_nccl_id_is_empty(&id)) {
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  return iree_hal_hip_nccl_channel_create(
      device->hip_symbols, device->nccl_symbols, &id, params.rank, params.count,
      device->host_allocator, out_channel);
}

static iree_status_t
iree_hal_hip_device_group_device_create_command_buffer_internal(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hip_device_group_device_commandbuffer_type_e type,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);

  iree_hal_command_buffer_t* buffers[IREE_HAL_MAX_QUEUES];
  memset(buffers, 0x00,
         sizeof(iree_hal_command_buffer_t*) * IREE_HAL_MAX_QUEUES);
  if (queue_affinity == 0) {
    queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  }
  queue_affinity = queue_affinity & ~(IREE_HAL_QUEUE_AFFINITY_ANY
                                      << device->num_physical_devices);

  iree_status_t status = iree_ok_status();
  uint32_t idx = 0;
  uint32_t cb_num = 0;
  iree_hal_queue_affinity_t affinity = queue_affinity;
  while (iree_status_is_ok(status) && affinity) {
    uint32_t nidx = iree_math_count_trailing_zeros_u64(affinity);
    idx += nidx;
    affinity >>= nidx + 1;
    status = IREE_HIP_RESULT_TO_STATUS(
        device->hip_symbols,
        hipCtxPushCurrent(device->device_contexts[idx].hip_context));
    if (!iree_status_is_ok(status)) {
      break;
    }
    switch (type) {
      case IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_STREAM:
        status = iree_status_join(
            status, iree_hal_hip_stream_command_buffer_create(
                        iree_hal_device_allocator(base_device),
                        device->hip_symbols, device->nccl_symbols,
                        device->device_contexts[idx].tracing_context, mode,
                        command_categories, binding_capacity,
                        device->device_contexts[idx].hip_dispatch_stream,
                        &device->block_pool, device->host_allocator,
                        (iree_hal_queue_affinity_t)1 << idx, &buffers[cb_num]));
        break;
      case IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH:
        status = iree_status_join(
            status,
            iree_hal_hip_graph_command_buffer_create(
                iree_hal_device_allocator(base_device), device->hip_symbols,
                device->device_contexts[idx].tracing_context,
                device->device_contexts[idx].hip_context, mode,
                command_categories, (iree_hal_queue_affinity_t)1 << idx,
                binding_capacity, &device->block_pool, device->host_allocator,
                &buffers[cb_num]));
        break;
    }
    status =
        IREE_HIP_RESULT_TO_STATUS(device->hip_symbols, hipCtxPopCurrent(NULL));
    ++idx;
    ++cb_num;
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    for (uint32_t i = 0; i < IREE_HAL_MAX_QUEUES; ++i) {
      if (buffers[i]) {
        iree_hal_resource_release(buffers[i]);
      }
    }
    return status;
  }

  return iree_hal_hip_device_group_command_buffer_create(
      device->host_allocator, cb_num, &buffers[0], device->device_allocator,
      mode, command_categories, queue_affinity, device->hip_symbols,
      device->num_physical_devices, device->device_contexts, binding_capacity,
      out_command_buffer);
}

iree_status_t iree_hal_hip_device_group_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_hal_hip_device_group_device_create_command_buffer_internal(
      base_device, mode, command_categories, queue_affinity, binding_capacity,
      IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_STREAM,
      out_command_buffer);
}

static iree_status_t iree_hal_hip_device_group_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  if (device->params.allow_inline_execution &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // The caller has indicated the command buffer can be executed as it is
    // recorded, implying that the command buffer cannot be reused and doesn't
    // need to be persisted. This lets us lower the execution delay as we can
    // directly route commands to a HIP stream and let it eagerly flush.
    return iree_hal_hip_device_group_device_create_command_buffer_internal(
        base_device, mode, command_categories, queue_affinity, binding_capacity,
        IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_STREAM,
        out_command_buffer);
  }
  switch (device->params.command_buffer_mode) {
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH:
      // TODO(indirect-cmd): when we can record indirect graphs we won't need
      // to use deferred command buffers - this is here to emulate indirect
      // command buffers.
      if (binding_capacity > 0) {
        return iree_hal_deferred_command_buffer_create(
            iree_hal_device_allocator(base_device), mode, command_categories,
            binding_capacity, &device->block_pool,
            iree_hal_device_host_allocator(base_device), queue_affinity,
            out_command_buffer);
      } else {
        return iree_hal_hip_device_group_device_create_command_buffer_internal(
            base_device, mode, command_categories, queue_affinity,
            binding_capacity,
            IREE_HAL_HIP_DEVICE_GROUP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH,
            out_command_buffer);
      }
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM:
      return iree_hal_deferred_command_buffer_create(
          iree_hal_device_allocator(base_device), mode, command_categories,
          binding_capacity, &device->block_pool,
          iree_hal_device_host_allocator(base_device), queue_affinity,
          out_command_buffer);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
  }
}

static iree_status_t iree_hal_hip_device_group_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implmeneted");
}

static iree_status_t iree_hal_hip_device_group_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hip_device_group_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  hipDevice_t devices[IREE_HAL_MAX_QUEUES];
  hipCtx_t contexts[IREE_HAL_MAX_QUEUES];
  for (size_t i = 0; i < device->num_physical_devices; ++i) {
    devices[i] = device->device_contexts[i].hip_device;
    contexts[i] = device->device_contexts[i].hip_context;
  }
  return iree_hal_hip_nop_executable_cache_create(
      identifier, device->hip_symbols, device->num_physical_devices,
      device->device_contexts, device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_hip_device_group_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  return iree_hal_hip_event_semaphore_create(
      initial_value, device->hip_symbols, device->timepoint_pool,
      &iree_hal_hip_device_group_device_issue_work_callback, device,
      device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hip_device_group_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement HIP semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_group_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Allocate from the pool; likely to fail in cases of virtual memory
  // exhaustion but the error may be deferred until a later synchronization.
  // If pools are not supported we allocate a buffer as normal from whatever
  // allocator is set on the device.
  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools &&
      !iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    uint64_t device_num = 0;
    if (queue_affinity) {
      device_num = iree_math_count_trailing_zeros_u64(queue_affinity);
      if (device_num > device->num_physical_devices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Device affinity out of range, maximum device is %d",
            device->num_physical_devices);
      }
    }
    status = iree_hal_hip_memory_pools_allocate(
        &device->memory_pools,
        device->device_contexts[device_num].hip_dispatch_stream, pool, params,
        allocation_size, out_buffer);
  } else {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(base_device), params, allocation_size,
        out_buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_group_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Schedule the buffer deallocation if we got it from a pool and otherwise
  // drop it on the floor and let it be freed when the buffer is released.
  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools) {
    uint64_t device_num = 0;
    if (queue_affinity) {
      device_num = iree_math_count_trailing_zeros_u64(queue_affinity);
      if (device_num > device->num_physical_devices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Device affinity out of range, maximum device is %d",
            device->num_physical_devices);
      }
    }

    status = iree_hal_hip_memory_pools_deallocate(
        &device->memory_pools,
        device->device_contexts[device_num].hip_dispatch_stream, buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

static iree_status_t iree_hal_hip_device_group_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_hip_device_group_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static void iree_hal_hip_device_group_device_collect_tracing_context(
    void* user_data) {
  iree_hal_stream_tracing_context_collect(
      (iree_hal_stream_tracing_context_t*)user_data);
}

typedef struct iree_hal_hip_device_group_semaphore_submit_callback_data_t {
  iree_hal_hip_device_group_device_t* device;
  iree_atomic_ref_count_t semaphore_signal_count;
  iree_host_size_t command_buffer_count;
  iree_hal_queue_affinity_t queue_affinity;
  iree_hal_command_buffer_t* const* command_buffers;
  iree_hal_buffer_binding_table_t const* binding_table;
  iree_hal_semaphore_list_t wait_semaphores_list;
  iree_hal_semaphore_list_t signal_semaphores_list;
  iree_hal_resource_set_t* resource_set;
} iree_hal_hip_device_group_semaphore_submit_callback_data_t;

static iree_status_t iree_hal_hip_device_group_execute_now(
    iree_hal_hip_device_group_semaphore_submit_callback_data_t* data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_EQ(iree_math_count_ones_u64(data->queue_affinity), 1,
                 "Cannot execute a command buffer on more than one queue");
  iree_hal_hip_device_group_device_t* device = data->device;
  iree_status_t status = iree_ok_status();

  uint32_t nidx = iree_math_count_trailing_zeros_u64(queue_affinity);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_RESULT_TO_STATUS(
              symbols,
              hipCtxPushCurrent(device->device_contexts[idx].hip_context)));

  // TODO(awoloszyn): Because of how hip works, if we only have a single
  // device in the device_group we could avoid waiting on any of these
  // semaphores, we are guaranteed to have waits, but if we want this
  // to work across multiple device/streams, we need these waits.
  for (iree_host_size_t i = 0;
       i < data->wait_semaphores_list.count && iree_status_is_ok(status); ++i) {
    iree_hal_hip_event_t* event;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_semaphore_get_hip_event(
                data->wait_semaphores_list.semaphores[i],
                data->wait_semaphores_list.payload_values[i], &event));
    // If we don't have an event, then we don't have to wait for it since it
    // has already been signaled on the host.
    if (!event) {
      continue;
    }

    status = IREE_HIP_RESULT_TO_STATUS(
        symbols,
        hipStreamWaitEvent(device->device_contexts[idx].hip_dispatch_stream,
                           iree_hal_hip_event_handle(event), 0));
    iree_hal_hip_event_release(event);
  }
  status = iree_status_join(
      status, IREE_HIP_RESULT_TO_STATUS(
                  symbols, hipStreamWaitEvent(
                               device->device_contexts[idx].hip_dispatch_stream,
                               iree_hal_hip_event_handle(event), 0)));
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // We have satisfied all of the waits.

  for (iree_host_size_t i = 0;
       i < data->command_buffer_count && iree_status_is_ok(status); ++i) {
    iree_hal_command_buffer_t* command_buffer = data->command_buffers[i];
    iree_hal_buffer_binding_table_t binding_table =
        data->binding_tables ? data->binding_tables[i]
                             : iree_hal_buffer_binding_table_empty();
    if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      iree_hal_command_buffer_mode_t mode =
          iree_hal_command_buffer_mode(command_buffer) |
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          // NOTE: we need to validate if a binding table is provided as the
          // bindings were not known when it was originally recorded.
          (iree_hal_buffer_binding_table_is_empty(binding_table)
               ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
               : 0);
      iree_hal_hip_device_group_device_create_stream_command_buffer(
          data->device, mode, command_buffer->allowed_categories,
          data->queue_affinity, 0, &stream_command_buffer);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(data->resource_set, 1,
                                           &stream_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_deferred_command_buffer_apply(
                  command_buffer, stream_command_buffer, binding_table));
    } else if (iree_hal_hip_stream_command_buffer_isa(command_buffer)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_resource_set_insert(data->resource_set, 1, &command_buffer));
    } else if (iree_hal_hip_graph_command_buffer_isa(command_buffer)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_resource_set_insert(data->resource_set, 1, &command_buffer));
      hipGraphExec_t exec = iree_hal_hip_graph_command_buffer_handle(
          (iree_hal_hip_graph_command_buffer_t*)command_buffer);
      status = IREE_HIP_RESULT_TO_STATUS(
          data->device->hip_symbols,
          hipGraphLaunch(exec,
                         device->device_contexts[nidx].hip_dispatch_stream));
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported command buffer type");
    }
  }

  for (iree_host_size_t i = 0; i < data->signal_semaphores_list.count; ++i) {
    iree_hal_hip_event_t* event;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_semaphore_get_hip_event(
                data->signal_semaphores_list.semaphores[i],
                data->signal_semaphores_list.payload_values[i], &event));
    if (!event) {
      return iree_make_status(IREE_STATUS_ABORTED, "The hip event is missing");
    }
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols,
        hipEventRecord(iree_hal_hip_event_handle(event),
                       device->device_contexts[nidx].hip_dispatch_stream));
    iree_hal_hip_event_release(event);
  }

  for (iree_host_size_t i = 0; i < data->signal_semaphores_list.count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_semaphore_notify_forward_progress_to(
                data->signal_semaphores_list.semaphores[i],
                data->signal_semaphores_list.payload_values[i]));
  }

  iree_hal_hip_event_t* event;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_hip_event_pool_acquire(
          data->device->device_contexts[nidx].device_event_pool, 1, &event));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_RESULT_TO_STATUS(
              symbols, hipEventRecord(
                           iree_hal_hip_event_handle(event),
                           device->device_contexts[nidx].hip_dispatch_stream)));

  iree_hal_hip_cleanup_thread_add_cleanup(
      device->cleanup_thread, event,
      &iree_hal_hip_device_group_device_complete_submission, data);

  status = IREE_HIP_RESULT_TO_STATUS(symbols, hipCtxPopCurrent(NULL));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_group_semaphore_submit_callback(
    void* user_context, iree_hal_semaphore_t* semaphore, iree_status_t status) {
  iree_hal_hip_device_group_semaphore_submit_callback_data_t* data =
      (iree_hal_hip_device_group_semaphore_submit_callback_data_t*)user_context;
  if (iree_atomic_ref_count_dec(&resource->ref_count) != 1) {
    return iree_status_ok();
  }
  if (!iree_ok_status(status)) {
    return status;
  }
  // Now the actual submit happens, as all semaphore have been satisfied
  // (by satisfied here, we specifically mean that the semaphore has been
  // scheduled, not necessarily completed)
  return iree_hal_hip_device_group_execute_now(data);
}

static iree_status_t iree_hal_hip_device_group_make_callback_data(
    iree_allocator_t host_allocator, iree_arena_block_pool_t block_pool,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables,
    iree_hal_hip_device_group_semaphore_submit_callback_data_t** out_data) {
  // Embed captured tables in the action allocation.
  iree_hal_hip_device_group_semaphore_submit_callback_data_t* callback_data =
      NULL;

  const iree_host_size_t wait_semaphore_list_size =
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores) +
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values);
  const iree_host_size_t signal_semaphore_list_size =
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores) +
      signal_semaphore_list.count *
          sizeof(*signal_semaphore_list.payload_values);
  const iree_host_size_t command_buffers_size =
      command_buffer_count * sizeof(*callback_data->command_buffers);
  iree_host_size_t binding_tables_size = 0;
  iree_host_size_t binding_table_elements_size = 0;
  if (binding_tables) {
    binding_tables_size = command_buffer_count * sizeof(*binding_tables);
    for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
      binding_table_elements_size +=
          binding_tables[i].count * sizeof(*binding_tables[i].bindings);
    }
  }
  const iree_host_size_t payload_size =
      command_buffers_size + binding_tables_size + binding_table_elements_size;
  const iree_host_size_t total_callback_size =
      sizeof(*callback_data) + wait_semaphore_list_size +
      signal_semaphore_list_size + payload_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, total_callback_size,
                                (void**)&callback_data));
  uint8_t* callback_ptr = (uint8_t*)callback_data + sizeof(*callback_data);

  // Copy wait list for later access.
  callback_data->wait_semaphore_list.count = wait_semaphore_list.count;
  callback_data->wait_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(callback_data->wait_semaphore_list.semaphores,
         wait_semaphore_list.semaphores,
         wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores));
  callback_data->wait_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + wait_semaphore_list.count *
                                     sizeof(*wait_semaphore_list.semaphores));
  memcpy(
      callback_data->wait_semaphore_list.payload_values,
      wait_semaphore_list.payload_values,
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values));
  callback_ptr += wait_semaphore_list_size;

  // Copy signal list for later access.
  callback_data->signal_semaphore_list.count = signal_semaphore_list.count;
  callback_data->signal_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(
      callback_data->signal_semaphore_list.semaphores,
      signal_semaphore_list.semaphores,
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores));
  callback_data->signal_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + signal_semaphore_list.count *
                                     sizeof(*signal_semaphore_list.semaphores));
  memcpy(callback_data->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         signal_semaphore_list.count *
             sizeof(*signal_semaphore_list.payload_values));
  callback_ptr += signal_semaphore_list_size;

  // Copy the execution resources for later access.
  callback_data->payload.execution.queue_affinity = queue_affinity;
  callback_data->payload.execution.count = command_buffer_count;
  callback_data->payload.execution.command_buffers =
      (iree_hal_command_buffer_t**)callback_ptr;
  memcpy(callback_data->payload.execution.command_buffers, command_buffers,
         command_buffers_size);
  callback_ptr += command_buffers_size;

  // Retain all command buffers and semaphores.
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &callback_data->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(callback_data->resource_set,
                                          wait_semaphore_list.count,
                                          wait_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(callback_data->resource_set,
                                          signal_semaphore_list.count,
                                          signal_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(
        callback_data->resource_set, command_buffer_count, command_buffers);
  }

  // Copy binding tables and retain all bindings.
  if (iree_status_is_ok(status) && binding_table_elements_size > 0) {
    callback_data->payload.execution.binding_tables =
        (iree_hal_buffer_binding_table_t*)callback_ptr;
    callback_ptr += binding_tables_size;
    iree_hal_buffer_binding_t* binding_element_ptr =
        (iree_hal_buffer_binding_t*)callback_ptr;
    for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
      iree_host_size_t element_count = binding_tables[i].count;
      iree_hal_buffer_binding_table_t* target_table =
          &callback_data->payload.execution.binding_tables[i];
      target_table->count = element_count;
      target_table->bindings = binding_element_ptr;
      memcpy((void*)target_table->bindings, binding_tables[i].bindings,
             element_count * sizeof(*binding_element_ptr));
      binding_element_ptr += element_count;

      // Bulk insert all bindings into the resource set. This will keep the
      // referenced buffers live until the callback_data has completed. Note
      // that if we fail here we need to clean up the resource set below before
      // returning.
      status = iree_hal_resource_set_insert_strided(
          callback_data->resource_set, element_count, target_table->bindings,
          offsetof(iree_hal_buffer_binding_t, buffer),
          sizeof(iree_hal_buffer_binding_t));
      if (!iree_status_is_ok(status)) break;
    }
  } else {
    callback_data->payload.execution.binding_tables = NULL;
  }
  *out_data = callback_data;
  return status;
}

static iree_status_t iree_hal_hip_device_group_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (queue_affinity == IREE_HAL_QUEUE_AFFINITY_ANY) {
    queue_affinity = 0x1;
  }

  uint64_t queue_affinity_mask =
      ((iree_hal_queue_affinity_t)1 << device->num_physical_devices);
  queue_affinity_mask = queue_affinity_mask | (queue_affinity_mask - 1);
  queue_affinity &= queue_affinity_mask;
  iree_hal_command_buffer_t** cbs = NULL;
  if (command_buffer_count) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(
                device->host_allocator,
                sizeof(iree_hal_command_buffer_t*) * command_buffer_count,
                (void**)&cbs));
  }

  uint32_t nidx = iree_math_count_trailing_zeros_u64(queue_affinity);
  queue_affinity = 1 << nidx;

  iree_hal_hip_device_group_semaphore_submit_callback_data_t* callback_data;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_device_group_make_callback_data(
              device->host_allocator, device->block_pool, queue_affinity,
              wait_semaphore_list, signal_semaphore_list, command_buffer_count,
              command_buffers, binding_tables, &callback_data));

  if (wait_semaphore_list.count == 0) {
    status = iree_hal_hip_device_group_execute_now(callback_data);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t i = 0;
       i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
    status = iree_hal_hip_semaphore_notify_work(
        wait_semaphore_list.semaphores[i],
        wait_semaphore_list.payload_values[i],
        &iree_hal_hip_device_group_semaphore_submit_callback, callback_data);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(device->host_allocator, callback_data);
  }

  return status;
}

static iree_status_t iree_hal_hip_device_group_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  uint64_t queue_affinity_mask =
      ((iree_hal_queue_affinity_t)1 << device->num_physical_devices);
  queue_affinity_mask = queue_affinity_mask | (queue_affinity_mask - 1);
  queue_affinity &= queue_affinity_mask;

  iree_status_t status = iree_ok_status();
  uint32_t idx = 0;
  while (iree_status_is_ok(status) && queue_affinity) {
    // Try to advance the deferred work queue.
    status =
        iree_status_join(status, iree_hal_deferred_work_queue_issue(
                                     device->device_contexts[idx].work_queue));
    ++idx;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_group_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast(base_device);
  return iree_hal_hip_semaphore_multi_wait(semaphore_list, wait_mode, timeout,
                                           &device->block_pool);
}

static iree_status_t iree_hal_hip_device_group_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_group_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_group_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

const iree_hal_hip_dynamic_symbols_t* iree_hal_hip_device_group_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_group_device_t* device =
      iree_hal_hip_device_group_device_cast_unsafe(base_device);
  return device->hip_symbols;
}

static const iree_hal_device_vtable_t iree_hal_hip_device_group_device_vtable =
    {
        .destroy = iree_hal_hip_device_group_device_destroy,
        .id = iree_hal_hip_device_group_device_id,
        .host_allocator = iree_hal_hip_device_group_device_host_allocator,
        .device_allocator = iree_hal_hip_device_group_device_allocator,
        .replace_device_allocator = iree_hal_hip_replace_device_allocator,
        .replace_channel_provider = iree_hal_hip_replace_channel_provider,
        .trim = iree_hal_hip_device_group_device_trim,
        .query_i64 = iree_hal_hip_device_group_device_query_i64,
        .create_channel = iree_hal_hip_device_group_device_create_channel,
        .create_command_buffer =
            iree_hal_hip_device_group_device_create_command_buffer,
        .create_event = iree_hal_hip_device_group_device_create_event,
        .create_executable_cache =
            iree_hal_hip_device_group_device_create_executable_cache,
        .import_file = iree_hal_hip_device_group_device_import_file,
        .create_semaphore = iree_hal_hip_device_group_device_create_semaphore,
        .query_semaphore_compatibility =
            iree_hal_hip_device_group_device_query_semaphore_compatibility,
        .queue_alloca = iree_hal_hip_device_group_device_queue_alloca,
        .queue_dealloca = iree_hal_hip_device_group_device_queue_dealloca,
        .queue_read = iree_hal_hip_device_group_device_queue_read,
        .queue_write = iree_hal_hip_device_group_device_queue_write,
        .queue_execute = iree_hal_hip_device_group_device_queue_execute,
        .queue_flush = iree_hal_hip_device_group_device_queue_flush,
        .wait_semaphores = iree_hal_hip_device_group_device_wait_semaphores,
        .profiling_begin = iree_hal_hip_device_group_device_profiling_begin,
        .profiling_flush = iree_hal_hip_device_group_device_profiling_flush,
        .profiling_end = iree_hal_hip_device_group_device_profiling_end,
};

static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_group_device_interface_vtable_t = {
        .destroy = iree_hal_hip_tracing_device_group_device_interface_destroy,
        .synchronize_native_event =
            iree_hal_hip_tracing_device_group_device_interface_synchronize_native_event,
        .create_native_event =
            iree_hal_hip_tracing_device_group_device_interface_create_native_event,
        .query_native_event =
            iree_hal_hip_tracing_device_group_device_interface_query_native_event,
        .event_elapsed_time =
            iree_hal_hip_tracing_device_group_device_interface_event_elapsed_time,
        .destroy_native_event =
            iree_hal_hip_tracing_device_group_device_interface_destroy_native_event,
        .record_native_event =
            iree_hal_hip_tracing_device_group_device_interface_record_native_event,
        .add_graph_event_record_node =
            iree_hal_hip_tracing_device_group_device_interface_add_graph_event_record_node,
};