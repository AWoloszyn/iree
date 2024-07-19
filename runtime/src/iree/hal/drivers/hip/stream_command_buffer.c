
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#define POOL_TYPE 2

#include "iree/hal/drivers/hip/stream_command_buffer.h"

#include <stdbool.h>

#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/native_executable.h"
#include "iree/hal/drivers/hip/pipeline_layout.h"
#include "iree/hal/drivers/hip/rccl_channel.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/tracing.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

#if POOL_TYPE == 2
#include "iree/base/circular_array.h"
#include "iree/base/hash_map.h"
#include "iree/base/interval_list.h"
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

#define MAX_STREAMS 32
#define MAX_GRAPH_HEADS 16
#define MAX_BUFFERED_COMMANDS 1024
#if POOL_TYPE == 1 || POOL_TYPE == 2
typedef struct iree_hal_hip_sync_event_list_t iree_hal_hip_sync_event_list_t;
typedef struct iree_hal_hip_sync_event_list_t {
  iree_hal_hip_sync_event_list_t* next;
  iree_hal_hip_event_t* event;
} iree_hal_hip_sync_event_list_t;
typedef struct dispatch {
  hipFunction_t f;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
} dispatch;
typedef struct copy_buffer {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
} copy_buffer;
typedef struct update_buffer {
  hipDeviceptr_t dst;
  void* src;
  size_t sizeBytes;
} update_buffer;
typedef struct fill_buffer {
  hipDeviceptr_t dst;
  uint64_t pattern;
  uint32_t pattern_length;
  size_t count;
} fill_buffer;

typedef struct iree_hal_hip_stream_command_buffer_t
    iree_hal_hip_stream_command_buffer_t;
typedef struct command command;

typedef struct command {
  union {
    dispatch dispatch;
    copy_buffer copy_buffer;
    update_buffer update_buffer;
    fill_buffer fill_buffer;
  };
  const char* debug_label;
  iree_status_t (*apply)(iree_hal_hip_stream_command_buffer_t*, command*,
                         uint32_t);
} command;
#endif

#if POOL_TYPE == 2
typedef struct iree_hal_hip_graph_node_t {
  command* command;
  iree_hash_map_t* children;
  iree_hash_map_t* parents;
  // To save some effort, scheduled_stream is going to be
  // ~actually_scheduled_stream. This way we can simply
  // check if its not null to see if its been scheduled.
  uint32_t scheduled_stream;
  // Track this here to save us time later.
  uint32_t num_parents_scheduled;
} iree_hal_hip_graph_node_t;

bool insert_nodes_into_circular_array(iree_hash_map_element_t* element,
                                      void* user_data) {
  iree_circular_array_t* array = (iree_circular_array_t*)(user_data);
  iree_hal_hip_graph_node_t* parent_node =
      (iree_hal_hip_graph_node_t*)iree_hash_map_element_get_key(element);

  iree_hal_hip_graph_node_t** out_item;
  IREE_ASSERT(iree_ok_status() ==
              iree_circular_array_push_back(array, (void**)&out_item));
  out_item[0] = parent_node;
  parent_node->num_parents_scheduled++;
  return true;
}

typedef struct pc {
  iree_hal_hip_graph_node_t* parent_node;
  iree_circular_array_t* array;
} pc;

bool print_children(iree_hash_map_element_t* element, void* user_data) {
  iree_hal_hip_graph_node_t* child_node =
      (iree_hal_hip_graph_node_t*)iree_hash_map_element_get_key(element);
  iree_hal_hip_graph_node_t* parent_node = ((pc*)user_data)->parent_node;
  fprintf(stderr, "\"%p\" -> \"%p\"\n", parent_node, child_node);
  iree_hal_hip_graph_node_t** out_item;
  IREE_ASSERT(iree_ok_status() ==
              iree_circular_array_push_back(((pc*)user_data)->array,
                                            (void**)&out_item));
  out_item[0] = child_node;
  return true;
}

void iree_hal_hip_graph_buffer_print(iree_allocator_t allocator,
                                     iree_hal_hip_graph_node_t** heads,
                                     iree_host_size_t num_heads) {
  iree_hash_map_t* inserted;
  iree_hash_map_create(allocator, 0, 4096, &inserted);

  iree_circular_array_t* array;
  iree_circular_array_create(allocator, 1024,
                             sizeof(iree_hal_hip_graph_node_t*), &array);

  for (size_t i = 0; i < num_heads; ++i) {
    iree_hal_hip_graph_node_t** node;
    iree_circular_array_push_back(array, (void**)&node);
    *node = heads[i];
  }
  fprintf(stderr, "digraph G {\n");

  while (iree_circular_array_size(array)) {
    iree_hal_hip_graph_node_t** node;
    iree_circular_array_front(array, (void**)&node);
    if (iree_hash_map_find(inserted, (uint64_t)node[0])) {
      iree_circular_array_pop_front(array);
      continue;
    }
    fprintf(stderr, "\"%p\" [label=\"%s [%p]\"];\n", node[0],
            node[0]->command->debug_label, node[0]);
    pc walk = {node[0], array};
    // First add all of our children to the circular array.
    iree_hash_map_walk(node[0]->children, print_children, &walk);
    iree_circular_array_pop_front(array);
    iree_hash_map_element_t* dummy;
    iree_hash_map_insert(inserted, (uint64_t)node[0], &dummy);
  }
  fprintf(stderr, "}\n");
}

typedef struct iree_hal_hip_buffer_interval_t {
  iree_hal_hip_graph_node_t* last_written_by;
} iree_hal_hip_buffer_interval_t;
#define IREE_READ_INTERVAL_MAX_READS 256
typedef struct iree_hal_hip_buffer_read_interval_t {
  iree_hal_hip_graph_node_t* last_read_by[IREE_READ_INTERVAL_MAX_READS];
  uint32_t last_read_count;
} iree_hal_hip_buffer_read_interval_t;
#endif

typedef struct iree_hal_hip_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols;

  // Per-stream HIP tracing context.
  iree_hal_hip_tracing_context_t* tracing_context;
  iree_hal_hip_tracing_context_event_list_t tracing_event_list;

  iree_hal_hip_queue_t* hip_queue;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need HIP to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  int32_t push_constants[IREE_HAL_HIP_MAX_PUSH_CONSTANT_COUNT];

  // The current bound descriptor sets.
  struct {
    hipDeviceptr_t bindings[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT];
    iree_host_size_t
        binding_sizes[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  } descriptor_sets[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_COUNT];
#if POOL_TYPE == 0
  iree_hal_hip_event_pool_t* event_pool;
  iree_hal_hip_sync_event_list_t* event_list;
  iree_hal_hip_event_t* last_sync_event;
#elif POOL_TYPE == 1
  int32_t* event_mem;
  int32_t event_num[MAX_STREAMS];
  command buffered_commands[1024];
  int32_t num_buffered_commands;
#elif POOL_TYPE == 2
  iree_interval_list_t* buffer_writes;
  iree_interval_list_t* buffer_reads;
  iree_hal_hip_graph_node_t*
      heads[MAX_GRAPH_HEADS];  // hopefully no more than this?
  int32_t num_graph_heads;
  int32_t total_elements;
#endif

  int32_t current_stream;
} iree_hal_hip_stream_command_buffer_t;

int32_t increment_stream(int32_t* current_stream) {
  if (*current_stream == MAX_STREAMS) {
    return (*current_stream) - 1;
  }
  return (*current_stream)++;
}

#if POOL_TYPE == 1
static iree_status_t iree_hal_hip_stream_command_buffer_apply_buffered_commands(
    iree_hal_hip_stream_command_buffer_t* cb) {
  iree_status_t status = iree_ok_status();
  for (int i = 0; i < cb->num_buffered_commands && iree_status_is_ok(status);
       ++i) {
    status = cb->buffered_commands[i].apply(cb, &cb->buffered_commands[i],
                                            cb->current_stream++);
  }
  cb->num_buffered_commands = 0;
  return status;
}

static command* iree_hal_hip_stream_command_buffer_get_next_buffered_command(
    iree_hal_hip_stream_command_buffer_t* cb) {
  IREE_ASSERT(cb->num_buffered_commands < MAX_BUFFERED_COMMANDS);
  return &cb->buffered_commands[cb->num_buffered_commands++];
}
#endif

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_stream_command_buffer_vtable;

static iree_hal_hip_stream_command_buffer_t*
iree_hal_hip_stream_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_stream_command_buffer_vtable);
  return (iree_hal_hip_stream_command_buffer_t*)base_value;
}
#if POOL_TYPE == 0
iree_status_t iree_hal_hip_stream_command_buffer_allocate_event(
    iree_hal_hip_stream_command_buffer_t* command_buffer,
    iree_hal_hip_event_t** out_event) {
  iree_hal_hip_sync_event_list_t* event = NULL;

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      command_buffer->host_allocator, sizeof(iree_hal_hip_sync_event_list_t),
      (void**)&event));
  IREE_RETURN_IF_ERROR(iree_hal_hip_event_pool_acquire(
      command_buffer->event_pool, 1, &event->event));

  event->next = command_buffer->event_list;
  command_buffer->event_list = event;
  out_event[0] = event->event;
  return iree_ok_status();
}
#endif
iree_status_t iree_hal_hip_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_hip_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_hal_hip_queue_t* queue,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hip_event_pool_t* event_pool,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hip_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->hip_symbols = hip_symbols;
  command_buffer->nccl_symbols = nccl_symbols;
  command_buffer->tracing_context = tracing_context;
  command_buffer->tracing_event_list.head = NULL;
  command_buffer->tracing_event_list.tail = NULL;
  command_buffer->hip_queue = queue;
#if POOL_TYPE == 0
  command_buffer->event_pool = event_pool;
  command_buffer->event_list = NULL;
#elif POOL_TYPE == 1
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipExtMallocWithFlags((void**)&command_buffer->event_mem,
                            sizeof(int32_t) * MAX_STREAMS + 1,
                            hipDeviceMallocFinegrained),
      "hipMalloc");
  memset(command_buffer->event_num, 0x00, sizeof(command_buffer->event_num));
#endif

  iree_status_t status = iree_ok_status();
#if POOL_TYPE == 2
  status = iree_interval_list_create(host_allocator,
                                     sizeof(iree_hal_hip_buffer_interval_t),
                                     &command_buffer->buffer_writes);

  if (iree_status_is_ok(status)) {
    status = iree_interval_list_create(
        host_allocator, sizeof(iree_hal_hip_buffer_read_interval_t),
        &command_buffer->buffer_reads);
  }
  command_buffer->num_graph_heads = 0;
#endif

  command_buffer->current_stream = 0;

  iree_arena_initialize(block_pool, &command_buffer->arena);

  status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  *out_command_buffer = &command_buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#if POOL_TYPE == 2
static void iree_hal_hip_stream_command_buffer_get_last_writes(
    iree_hal_hip_stream_command_buffer_t* cb, hipDeviceptr_t base,
    iree_host_size_t size, iree_interval_t** write_begin,
    iree_interval_t** write_end) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_interval_list_find(cb->buffer_writes, (uint64_t)base, (uint64_t)size,
                          write_begin, write_end);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_hip_stream_command_buffer_get_last_reads(
    iree_hal_hip_stream_command_buffer_t* cb, hipDeviceptr_t base,
    iree_host_size_t size, iree_interval_t** read_begin,
    iree_interval_t** read_end) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_interval_list_find(cb->buffer_reads, (uint64_t)base, (uint64_t)size,
                          read_begin, read_end);
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_hip_buffer_interval_t* iree_hal_hip_write_interval_get_data(
    iree_interval_t* interval) {
  return (iree_hal_hip_buffer_interval_t*)iree_interval_get_data(interval);
}

static iree_hal_hip_buffer_read_interval_t* iree_hal_hip_read_interval_get_data(
    iree_interval_t* interval) {
  return (iree_hal_hip_buffer_read_interval_t*)iree_interval_get_data(interval);
}

static iree_status_t iree_hal_hip_stream_command_buffer_register_write(
    iree_hal_hip_stream_command_buffer_t* cb, hipDeviceptr_t base,
    iree_host_size_t size, iree_hal_hip_graph_node_t* node) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_interval_t* interval;
  iree_status_t status = iree_interval_list_insert(
      cb->buffer_writes, (uint64_t)base, size, &interval);
  IREE_ASSERT(iree_status_is_ok(status));  // TODO: error handling
  iree_hal_hip_write_interval_get_data(interval)->last_written_by = node;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_register_read(
    iree_hal_hip_stream_command_buffer_t* cb, hipDeviceptr_t base,
    iree_host_size_t size, iree_hal_hip_graph_node_t* node) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_interval_t* interval_begin;
  iree_interval_t* interval_end;
  iree_status_t status = iree_interval_list_insert_no_overwrite(
      cb->buffer_reads, (uint64_t)base, size, &interval_begin, &interval_end);
  IREE_ASSERT(iree_status_is_ok(status));  // TODO: error handling
  while (interval_begin != interval_end) {
    iree_hal_hip_buffer_read_interval_t* interval =
        iree_hal_hip_read_interval_get_data(interval_begin);
    IREE_ASSERT(interval->last_read_count < IREE_READ_INTERVAL_MAX_READS);
    interval->last_read_by[interval->last_read_count++] = node;
    interval_begin = iree_interval_next(interval_begin);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_add_buffer_access(
    iree_hal_hip_stream_command_buffer_t* cb, iree_hal_hip_graph_node_t* node,
    bool is_write, hipDeviceptr_t base, iree_host_size_t size,
    iree_host_size_t* num_parents) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (is_write) {
    iree_interval_t* begin;
    iree_interval_t* end;
    iree_hal_hip_stream_command_buffer_get_last_reads(cb, base, size, &begin,
                                                      &end);
    while (begin != end) {
      iree_hal_hip_buffer_read_interval_t* read =
          iree_hal_hip_read_interval_get_data(begin);
      for (iree_host_size_t i = 0; i < read->last_read_count; ++i) {
        iree_hash_map_element_t* inserted = NULL;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hash_map_try_insert(
                    node->parents, (uint64_t)read->last_read_by[i], &inserted));
        (*num_parents)++;
        if (inserted) {
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hash_map_try_insert(read->last_read_by[i]->children,
                                           (uint64_t)node, &inserted));
        }
      }
      begin = iree_interval_next(begin);
    }
    iree_interval_list_erase(cb->buffer_reads, (uint64_t)base, size);
  }

  iree_interval_t* begin;
  iree_interval_t* end;
  iree_hal_hip_stream_command_buffer_get_last_writes(cb, base, size, &begin,
                                                     &end);
  while (begin != end) {
    iree_hal_hip_buffer_interval_t* write =
        iree_hal_hip_write_interval_get_data(begin);

    iree_hash_map_element_t* inserted = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hash_map_try_insert(
                node->parents, (uint64_t)write->last_written_by, &inserted));
    if (inserted) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hash_map_try_insert(write->last_written_by->children,
                                       (uint64_t)node, &inserted));
      (*num_parents)++;
    }
    begin = iree_interval_next(begin);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_hip_stream_command_buffer_add_root_node(
    iree_hal_hip_stream_command_buffer_t* cb, iree_hal_hip_graph_node_t* node) {
  IREE_ASSERT(cb->num_graph_heads < MAX_GRAPH_HEADS - 1);
  cb->heads[cb->num_graph_heads++] = node;
}

void iree_hal_hip_stream_command_buffer_destroy_graph_node(
    iree_hal_hip_stream_command_buffer_t* cb, iree_hal_hip_graph_node_t* node) {
  if (node->children) {
    iree_hash_map_destroy(node->children);
  }
  if (node->parents) {
    iree_hash_map_destroy(node->children);
  }
  iree_allocator_free(cb->host_allocator, node->command);
}

iree_status_t iree_hal_hip_stream_command_buffer_create_graph_node(
    iree_hal_hip_stream_command_buffer_t* cb,
    iree_hal_hip_graph_node_t** node) {
  iree_hal_hip_graph_node_t* graph_node;

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(cb->host_allocator,
                                             sizeof(iree_hal_hip_graph_node_t),
                                             (void**)&graph_node));
  iree_status_t status = iree_allocator_malloc(
      cb->host_allocator, sizeof(command), (void**)&graph_node->command);
  if (iree_status_is_ok(status)) {
    status =
        iree_hash_map_create(cb->host_allocator, 0, 8, &graph_node->parents);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hash_map_create(cb->host_allocator, 0, 8, &graph_node->children);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_hip_stream_command_buffer_destroy_graph_node(cb, graph_node);
  }
  *node = graph_node;
  return status;
}

#endif

static void iree_hal_hip_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_tracing_free(command_buffer->tracing_context,
                            &command_buffer->tracing_event_list);

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);

#if POOL_TYPE == 0
  while (command_buffer->event_list) {
    iree_hal_hip_sync_event_list_t* list = command_buffer->event_list;
    iree_hal_hip_event_release(list->event);
    command_buffer->event_list = list->next;
    iree_allocator_free(host_allocator, list);
  }
#elif POOL_TYPE == 1
  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "FreeAsync");
  IREE_HIP_IGNORE_ERROR(command_buffer->hip_symbols,
                        hipFreeAsync(command_buffer->event_mem,
                                     iree_hal_hip_queue_get_stream(
                                         command_buffer->hip_queue, 0)));
  IREE_TRACE_ZONE_END(z1);
#elif POOL_TYPE == 2

  if (command_buffer->buffer_writes) {
    iree_interval_list_free(command_buffer->buffer_writes);
  }
#endif
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hip_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_hip_stream_command_buffer_vtable);
}

void iree_hal_hip_stream_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  if (!command_buffer->tracing_context) {
    return;
  }

  iree_hal_hip_tracing_notify_submitted(command_buffer->tracing_context,
                                        &command_buffer->tracing_event_list);
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_hip_stream_command_buffer_flush_collectives(
    iree_hal_hip_stream_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_ASSERT(false, "Not handling this yet");
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hip_nccl_submit_batch(
      command_buffer->nccl_symbols, command_buffer->tracing_context,
      &command_buffer->tracing_event_list, &command_buffer->collective_batch,
      command_buffer->hip_queue);
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_HIP_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      command_buffer->hip_queue,
      /*file_name=*/NULL, 0, /*line=*/0, "iree_hal_hip_stream_command_buffer",
      strlen("iree_hal_hip_stream_command_buffer"),
      /*name=*/NULL, 0);

#if POOL_TYPE == 0
  // Create an event so that every subsequent stream can wait on it.
  iree_hal_hip_event_t* event;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_allocate_event(command_buffer,
                                                            &event));

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipEventRecord(
          iree_hal_hip_event_handle(event),
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0)),
      "hipEventRecord");
  command_buffer->last_sync_event = event;
#endif

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
#if POOL_TYPE == 0
  // If we have more than one stream active, have stream0 wait on all of them
  for (int32_t i = 1; i < command_buffer->current_stream; ++i) {
    iree_hal_hip_event_t* event;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_stream_command_buffer_allocate_event(command_buffer,
                                                              &event));

    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipEventRecord(
            iree_hal_hip_event_handle(event),
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i)),
        "hipEventRecord");

    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
            iree_hal_hip_event_handle(event), 0),
        "hipStreamWaitEvent");
  }
#elif POOL_TYPE == 1
  int32_t last_active_streams = command_buffer->current_stream;
  command_buffer->current_stream = 0;
  if (last_active_streams == 0 || (command_buffer->num_buffered_commands <= 1 &&
                                   last_active_streams == 1)) {
    iree_hal_hip_stream_command_buffer_apply_buffered_commands(command_buffer);
  } else {
    const int32_t num_buffered_commands = command_buffer->num_buffered_commands;
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWriteValue32(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
            &command_buffer->event_mem[0], ++command_buffer->event_num[0], 0),
        "hipStreamWriteValue32");
    for (int32_t i = 1; i < num_buffered_commands; i++) {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipStreamWaitValue32(
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i),
              &command_buffer->event_mem[0], command_buffer->event_num[0],
              hipStreamWaitValueEq, 0xFFFFFFFF),
          "hipStreamWaitValue32");
    }
    iree_hal_hip_stream_command_buffer_apply_buffered_commands(command_buffer);
    for (int32_t i = 1; i < num_buffered_commands; i++) {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipStreamWriteValue32(
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i),
              &command_buffer->event_mem[i + 1], ++command_buffer->event_num[i],
              0),
          "hipStreamWriteValue32");
      if (i % 2 == 0) {
        uint64_t waitval;
        memcpy(&waitval, command_buffer->event_num + i - 1, sizeof(uint64_t));
        IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
            z0, command_buffer->hip_symbols,
            hipStreamWaitValue64(
                iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
                &command_buffer->event_mem[i], waitval, hipStreamWaitValueEq,
                0xFFFFFFFFFFFFFFFF),
            "hipStreamWaitValue64");
      } else if (i == command_buffer->current_stream - 1) {
        IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
            z0, command_buffer->hip_symbols,
            hipStreamWaitValue32(
                iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
                &command_buffer->event_mem[i + 1], command_buffer->event_num[i],
                hipStreamWaitValueEq, 0xFFFFFFFF),
            "hipStreamWaitValue32");
      }
    }
  }
#elif POOL_TYPE == 2
  // if(command_buffer->total_elements > 1) {
  //   iree_hal_hip_graph_buffer_print(command_buffer->host_allocator,
  //   command_buffer->heads, command_buffer->num_graph_heads);
  // }

  iree_circular_array_t* array;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_circular_array_create(command_buffer->host_allocator, 1024,
                                 sizeof(iree_hal_hip_graph_node_t*), &array));

  for (size_t i = 0; i < command_buffer->num_graph_heads; ++i) {
    iree_hal_hip_graph_node_t** node;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_circular_array_push_back(array, (void**)&node));
    *node = command_buffer->heads[i];
  }

  while (iree_circular_array_size(array)) {
    iree_hal_hip_graph_node_t** node;
    iree_circular_array_front(array, (void**)&node);
    // All of our parents have been scheduled, we can too.
    if (!node[0]->scheduled_stream && iree_hash_map_size(node[0]->parents) ==
                                          node[0]->num_parents_scheduled) {
      // First add all of our children to the circular array.
      iree_hash_map_walk(node[0]->children, insert_nodes_into_circular_array,
                         array);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, node[0]->command->apply(command_buffer, node[0]->command, 0));
      node[0]->scheduled_stream = ~0;

      if (command_buffer->total_elements > 1) {
        fprintf(stderr, "%p\n", node[0]);
      }
    }
    iree_circular_array_pop_front(array);
  }
  iree_circular_array_free(array);
#endif
  command_buffer->current_stream = 0;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  // NOTE: the resource set may contain resources we need to drop as we don't
  //       need to keep them live any longer than it takes to schedule the
  //       operations. In a real command buffer we would be this stream command
  //       buffer is strictly used to perform inline execution/replay of
  //       deferred command buffers that are retaining the resources already.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_HIP_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 command_buffer->hip_queue);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HIP_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      command_buffer->hip_queue, location ? location->file.data : NULL,
      location ? location->file.size : 0, location ? location->line : 0,
      /*func_name=*/NULL, 0, label.data, label.size);
}

static void iree_hal_hip_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HIP_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 command_buffer->hip_queue);
}

static iree_status_t iree_hal_hip_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
#if POOL_TYPE == 0

  // If we have more than one stream active, have stream0 wait on all of them
  for (int32_t i = 1; i < command_buffer->current_stream; ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "Allocate_Event");
    iree_hal_hip_event_t* event;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_stream_command_buffer_allocate_event(command_buffer,
                                                              &event));
    IREE_TRACE_ZONE_END(z1);

    IREE_TRACE_ZONE_BEGIN_NAMED(z2, "hipEventRecord");
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipEventRecord(
            iree_hal_hip_event_handle(event),
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i)),
        "hipEventRecord");
    IREE_TRACE_ZONE_END(z2);

    IREE_TRACE_ZONE_BEGIN_NAMED(z3, "hipStreamWaitEvent");
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
            iree_hal_hip_event_handle(event), 0),
        "hipStreamWaitEvent");
    IREE_TRACE_ZONE_END(z3);
  }

  // Create an event so that every subsequent stream can wait on it.
  iree_hal_hip_event_t* event;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_allocate_event(command_buffer,
                                                            &event));

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipEventRecord(
          iree_hal_hip_event_handle(event),
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0)),
      "hipEventRecord");
  command_buffer->last_sync_event = event;
#elif POOL_TYPE == 1
  int32_t last_active_streams = command_buffer->current_stream;
  command_buffer->current_stream = 0;
  if (last_active_streams == 0 || (command_buffer->num_buffered_commands <= 1 &&
                                   last_active_streams == 1)) {
    iree_hal_hip_stream_command_buffer_apply_buffered_commands(command_buffer);
  } else {
    const int32_t num_buffered_commands = command_buffer->num_buffered_commands;
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWriteValue32(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
            &command_buffer->event_mem[0], ++command_buffer->event_num[0], 0),
        "hipStreamWriteValue32");
    for (int32_t i = 1; i < num_buffered_commands; i++) {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipStreamWaitValue32(
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i),
              &command_buffer->event_mem[0], command_buffer->event_num[0],
              hipStreamWaitValueEq, 0xFFFFFFFF),
          "hipStreamWaitValue32");
    }
    iree_hal_hip_stream_command_buffer_apply_buffered_commands(command_buffer);
    for (int32_t i = 1; i < num_buffered_commands; i++) {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipStreamWriteValue32(
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, i),
              &command_buffer->event_mem[i + 1], ++command_buffer->event_num[i],
              0),
          "hipStreamWriteValue32");
      if (i % 2 == 0) {
        uint64_t waitval;
        memcpy(&waitval, command_buffer->event_num + i - 1, sizeof(uint64_t));
        IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
            z0, command_buffer->hip_symbols,
            hipStreamWaitValue64(
                iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
                &command_buffer->event_mem[i], waitval, hipStreamWaitValueEq,
                0xFFFFFFFFFFFFFFFF),
            "hipStreamWaitValue64");
      } else if (i == command_buffer->current_stream - 1) {
        IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
            z0, command_buffer->hip_symbols,
            hipStreamWaitValue32(
                iree_hal_hip_queue_get_stream(command_buffer->hip_queue, 0),
                &command_buffer->event_mem[i + 1], command_buffer->event_num[i],
                hipStreamWaitValueEq, 0xFFFFFFFF),
            "hipStreamWaitValue32");
      }
    }
  }
#endif

  command_buffer->current_stream = 0;

  IREE_RETURN_IF_ERROR(
      iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Nothing to do for barriers between memory operations or dispatches--HIP
  // stream semantics guarantees execution and memory visibility in program
  // order.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hip_stream_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  // We could mark the memory as invalidated so that if managed HIP does not
  // try to copy it back to the host.
  return iree_ok_status();
}

iree_status_t iree_hal_hip_stream_command_buffer_apply_deferred_fill_buffer(
    iree_hal_hip_stream_command_buffer_t* command_buffer, command* command,
    uint32_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (command->fill_buffer.pattern_length) {
    case 4: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD32Async(
              command->fill_buffer.dst,
              *(const uint32_t*)(&command->fill_buffer.pattern),
              command->fill_buffer.count,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream)),
          "hipMemsetD32Async");
      break;
    }
    case 2: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD16Async(
              command->fill_buffer.dst,
              *(const uint16_t*)(&command->fill_buffer.pattern),
              command->fill_buffer.count,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream)),
          "hipMemsetD16Async");
      break;
    }
    case 1: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD8Async(
              command->fill_buffer.dst,
              *(const uint8_t*)(&command->fill_buffer.pattern),
              command->fill_buffer.count,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream)),
          "hipMemsetD8Async");
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer + target_offset;
  size_t num_elements = target_ref.length / pattern_length;

#if POOL_TYPE == 0
  if (IREE_UNLIKELY(command_buffer->current_stream > 0)) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                          command_buffer->current_stream),
            iree_hal_hip_event_handle(command_buffer->last_sync_event), 0),
        "hipStreamWaitEvent");
  }
  switch (pattern_length) {
    case 4: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD32Async(
              dst, *(const uint32_t*)(pattern), num_elements,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                            command_buffer->current_stream++)),
          "hipMemsetD32Async");
      break;
    }
    case 2: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD16Async(
              dst, *(const uint16_t*)(pattern), num_elements,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                            command_buffer->current_stream++)),
          "hipMemsetD16Async");
      break;
    }
    case 1: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hip_symbols,
          hipMemsetD8Async(
              dst, *(const uint8_t*)(pattern), num_elements,
              iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                            command_buffer->current_stream++)),
          "hipMemsetD8Async");
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }
#elif POOL_TYPE == 1
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unsupported fill pattern length");
  }
  command* cmd = iree_hal_hip_stream_command_buffer_get_next_buffered_command(
      command_buffer);
  cmd->fill_buffer = (fill_buffer){dst, 0, pattern_length, num_elements};
  memcpy(&cmd->fill_buffer.pattern, pattern, pattern_length);
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_fill_buffer;
  cmd->debug_label = "fill_buffer";
#elif POOL_TYPE == 2
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unsupported fill pattern length");
  }
  iree_hal_hip_graph_node_t* node;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_create_graph_node(command_buffer,
                                                               &node));

  command* cmd = node->command;
  cmd->fill_buffer = (fill_buffer){dst, 0, pattern_length, num_elements};
  memcpy(&cmd->fill_buffer.pattern, pattern, pattern_length);
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_fill_buffer;
  cmd->debug_label = "fill_buffer";
  iree_host_size_t num_parents = 0;
  iree_hal_hip_stream_command_buffer_add_buffer_access(
      command_buffer, node, true, dst, pattern_length * num_elements,
      &num_parents);
  // TODO proper error handling
  IREE_ASSERT(
      iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_write(
          command_buffer, dst, pattern_length * num_elements, node)));
  if (!num_parents) {
    iree_hal_hip_stream_command_buffer_add_root_node(command_buffer, node);
  }
  command_buffer->total_elements++;
#endif

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_hip_stream_command_buffer_apply_deferred_update_buffer(
    iree_hal_hip_stream_command_buffer_t* command_buffer, command* command,
    uint32_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyHtoDAsync(
          command->update_buffer.dst, command->update_buffer.src,
          command->update_buffer.sizeBytes,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream)),
      "hipMemcpyHtoDAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because HIP memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                                (void**)&storage));
    memcpy(storage, src, target_ref.length);
    src = storage;
  }

  // Issue the copy using the scratch memory as the source.
  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer +
                       iree_hal_buffer_byte_offset(target_ref.buffer) +
                       target_ref.offset;

#if POOL_TYPE == 0
  if (IREE_UNLIKELY(command_buffer->current_stream > 0)) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                          command_buffer->current_stream),
            iree_hal_hip_event_handle(command_buffer->last_sync_event), 0),
        "hipStreamWaitEvent");
  }

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyHtoDAsync(
          dst, (void*)src, target_ref.length,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                        command_buffer->current_stream++)),
      "hipMemcpyHtoDAsync");

#elif POOL_TYPE == 1
  command* cmd = iree_hal_hip_stream_command_buffer_get_next_buffered_command(
      command_buffer);
  cmd->update_buffer = (update_buffer){dst, (void*)src, target_ref.length};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_update_buffer;
  cmd->debug_label = "fill_buffer";
#elif POOL_TYPE == 2
  iree_hal_hip_graph_node_t* node;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_create_graph_node(command_buffer,
                                                               &node));

  command* cmd = node->command;
  cmd->update_buffer = (update_buffer){dst, (void*)src, target_ref.length};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_update_buffer;
  cmd->debug_label = "fill_buffer";
  iree_host_size_t num_parents = 0;
  iree_hal_hip_stream_command_buffer_add_buffer_access(
      command_buffer, node, true, dst, target_ref.length, &num_parents);
  // TODO proper error handling
  IREE_ASSERT(
      iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_write(
          command_buffer, dst, target_ref.length, node)));
  if (!num_parents) {
    iree_hal_hip_stream_command_buffer_add_root_node(command_buffer, node);
  }
  command_buffer->total_elements++;
#endif

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_hip_stream_command_buffer_apply_deferred_copy_buffer(
    iree_hal_hip_stream_command_buffer_t* command_buffer, command* command,
    uint32_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyAsync(
          command->copy_buffer.dst, command->copy_buffer.src,
          command->copy_buffer.sizeBytes, command->copy_buffer.kind,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream)),
      "hipMemcpyAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  hipDeviceptr_t target_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hipDeviceptr_t source_device_buffer = iree_hal_hip_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  hipDeviceptr_t dst = (uint8_t*)target_device_buffer + target_offset;
  hipDeviceptr_t src = (uint8_t*)source_device_buffer + source_offset;

#if POOL_TYPE == 0
  if (IREE_UNLIKELY(command_buffer->current_stream > 0)) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                          command_buffer->current_stream),
            iree_hal_hip_event_handle(command_buffer->last_sync_event), 0),
        "hipStreamWaitEvent");
  }

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hip_symbols,
      hipMemcpyAsync(
          dst, src, target_ref.length, hipMemcpyDeviceToDevice,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                        command_buffer->current_stream++)),
      "hipMemcpyAsync");

#elif POOL_TYPE == 1
  command* cmd = iree_hal_hip_stream_command_buffer_get_next_buffered_command(
      command_buffer);
  cmd->copy_buffer =
      (copy_buffer){dst, src, target_ref.length, hipMemcpyDeviceToDevice};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_copy_buffer;
  cmd->debug_label = "copy_buffer";
#elif POOL_TYPE == 2
  iree_hal_hip_graph_node_t* node;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_create_graph_node(command_buffer,
                                                               &node));

  command* cmd = node->command;
  cmd->copy_buffer =
      (copy_buffer){dst, src, target_ref.length, hipMemcpyDeviceToDevice};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_copy_buffer;
  cmd->debug_label = "copy_buffer";
  iree_host_size_t num_parents = 0;
  iree_hal_hip_stream_command_buffer_add_buffer_access(
      command_buffer, node, false, src, target_ref.length, &num_parents);
  iree_hal_hip_stream_command_buffer_add_buffer_access(
      command_buffer, node, true, dst, target_ref.length, &num_parents);
  // TODO proper error handling
  IREE_ASSERT(
      iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_write(
          command_buffer, dst, target_ref.length, node)));
  IREE_ASSERT(
      iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_read(
          command_buffer, src, target_ref.length, node)));
  if (!num_parents) {
    iree_hal_hip_stream_command_buffer_add_root_node(command_buffer, node);
  }
  command_buffer->total_elements++;
#endif

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_binding_t send_binding = {
      .buffer = send_ref.buffer,
      .offset = send_ref.offset,
      .length = send_ref.length,
  };
  iree_hal_buffer_binding_t recv_binding = {
      .buffer = recv_ref.buffer,
      .offset = recv_ref.offset,
      .length = recv_ref.length,
  };
  iree_status_t status = iree_hal_collective_batch_append(
      &command_buffer->collective_batch, channel, op, param, send_binding,
      recv_binding, element_count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constants[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count, const iree_hal_buffer_ref_t* bindings) {
  if (binding_count > IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "exceeded available binding slots for push "
        "descriptor set #%" PRIu32 "; requested %" PRIhsz " vs. maximal %d",
        set, binding_count, IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }

  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  hipDeviceptr_t* current_bindings =
      command_buffer->descriptor_sets[set].bindings;
  iree_host_size_t* current_sizes =
      command_buffer->descriptor_sets[set].binding_sizes;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings[i];
    hipDeviceptr_t device_ptr = NULL;
    iree_host_size_t size = 0;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));

      hipDeviceptr_t device_buffer = iree_hal_hip_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = (uint8_t*)device_buffer + offset + binding->offset;
      size = binding->length;
      if (size == IREE_WHOLE_BUFFER) {
        size =
            iree_hal_buffer_allocation_size(binding->buffer) - binding->offset;
      }
    }
    current_bindings[binding->ordinal] = device_ptr;
    current_sizes[binding->ordinal] = size;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_hip_stream_command_buffer_apply_deferred_dispatch(
    iree_hal_hip_stream_command_buffer_t* command_buffer, command* command,
    uint32_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      command_buffer->hip_symbols,
      hipModuleLaunchKernel(
          command->dispatch.f, command->dispatch.gridDimX,
          command->dispatch.gridDimY, command->dispatch.gridDimZ,
          command->dispatch.blockDimX, command->dispatch.blockDimY,
          command->dispatch.blockDimZ, command->dispatch.sharedMemBytes,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue, stream),
          command->dispatch.kernelParams, NULL),
      "hipModuleLaunchKernel");

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_hip_stream_command_buffer_t* command_buffer =
      iree_hal_hip_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_hip_kernel_info_t kernel_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  // IREE_HIP_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
  //     command_buffer->tracing_context, &command_buffer->tracing_event_list,
  //     command_buffer->hip_stream, kernel_info.source_filename.data,
  //     kernel_info.source_filename.size, kernel_info.source_line,
  //     kernel_info.function_name.data, kernel_info.function_name.size,
  //     /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  iree_hal_hip_dispatch_layout_t dispatch_layout =
      iree_hal_hip_pipeline_layout_dispatch_layout(kernel_info.layout);

  // The total number of descriptors across all descriptor sets.
  iree_host_size_t descriptor_count = dispatch_layout.total_binding_count;
  // The total number of push constants.
  iree_host_size_t push_constant_count = dispatch_layout.push_constant_count;
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count = descriptor_count + push_constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  // Each kernel_params[i] is itself a pointer to the corresponding
  // element at the *second* inline allocation at the end of the current
  // segment.
  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;

  // Set up kernel arguments to point to the payload slots.
  hipDeviceptr_t* payload_ptr =
      (hipDeviceptr_t*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }

  // Copy descriptors from all sets to the end of the current segment for later
  // access.
  iree_host_size_t set_count = dispatch_layout.set_layout_count;
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_hip_descriptor_set_layout_binding_count(
            iree_hal_hip_pipeline_layout_descriptor_set_layout(
                kernel_info.layout, i));
    iree_host_size_t index =
        iree_hal_hip_pipeline_layout_base_binding_index(kernel_info.layout, i);
    memcpy(payload_ptr + index, command_buffer->descriptor_sets[i].bindings,
           binding_count * sizeof(hipDeviceptr_t));
  }

  // Append the push constants to the kernel arguments.
  iree_host_size_t base_index = dispatch_layout.push_constant_base_index;
  // As commented in the above, what each kernel parameter points to is a
  // hipDeviceptr_t, which as the size of a pointer on the target machine. we
  // are just storing a 32-bit value for the push constant here instead. So we
  // must process one element each type, for 64-bit machines.
  for (iree_host_size_t i = 0; i < push_constant_count; i++) {
    *((uint32_t*)params_ptr[base_index + i]) =
        command_buffer->push_constants[i];
  }

#if POOL_TYPE == 0
  if (IREE_UNLIKELY(command_buffer->current_stream > 0)) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, command_buffer->hip_symbols,
        hipStreamWaitEvent(
            iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                          command_buffer->current_stream),
            iree_hal_hip_event_handle(command_buffer->last_sync_event), 0),
        "hipStreamWaitEvent");
  }

  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      command_buffer->hip_symbols,
      hipModuleLaunchKernel(
          kernel_info.function, workgroup_x, workgroup_y, workgroup_z,
          kernel_info.block_size[0], kernel_info.block_size[1],
          kernel_info.block_size[2], kernel_info.shared_memory_size,
          iree_hal_hip_queue_get_stream(command_buffer->hip_queue,
                                        command_buffer->current_stream++),
          params_ptr, NULL),
      "hipModuleLaunchKernel");

  iree_status_t status = iree_ok_status();
#elif POOL_TYPE == 1
  command* cmd = iree_hal_hip_stream_command_buffer_get_next_buffered_command(
      command_buffer);
  cmd->dispatch = (dispatch){kernel_info.function,
                             workgroup_x,
                             workgroup_y,
                             workgroup_z,
                             kernel_info.block_size[0],
                             kernel_info.block_size[1],
                             kernel_info.block_size[2],
                             kernel_info.shared_memory_size,
                             params_ptr};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_dispatch;
  cmd->debug_label = "dispatch";
#elif POOL_TYPE == 2
  iree_hal_hip_graph_node_t* node;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_stream_command_buffer_create_graph_node(command_buffer,
                                                               &node));

  command* cmd = node->command;
  cmd->dispatch = (dispatch){kernel_info.function,
                             workgroup_x,
                             workgroup_y,
                             workgroup_z,
                             kernel_info.block_size[0],
                             kernel_info.block_size[1],
                             kernel_info.block_size[2],
                             kernel_info.shared_memory_size,
                             params_ptr};
  cmd->apply = &iree_hal_hip_stream_command_buffer_apply_deferred_dispatch;
  cmd->debug_label = "dispatch";

  iree_host_size_t num_parents = 0;
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    const iree_hal_descriptor_set_layout_t* layout =
        iree_hal_hip_pipeline_layout_descriptor_set_layout(kernel_info.layout,
                                                           i);
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_hip_descriptor_set_layout_binding_count(layout);
    iree_hal_descriptor_flags_t* flags =
        iree_hal_hip_descriptor_set_layout_binding_flags(layout);
    for (iree_host_size_t j = 0; j < binding_count; ++j) {
      hipDeviceptr_t ptr = command_buffer->descriptor_sets[i].bindings[j];
      iree_host_size_t sz = command_buffer->descriptor_sets[i].binding_sizes[j];
      iree_hal_hip_stream_command_buffer_add_buffer_access(
          command_buffer, node, flags[j] == IREE_HAL_DESCRIPTOR_FLAG_NONE, ptr,
          sz, &num_parents);
    }
  }
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    const iree_hal_descriptor_set_layout_t* layout =
        iree_hal_hip_pipeline_layout_descriptor_set_layout(kernel_info.layout,
                                                           i);
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_hip_descriptor_set_layout_binding_count(layout);
    iree_hal_descriptor_flags_t* flags =
        iree_hal_hip_descriptor_set_layout_binding_flags(layout);
    for (iree_host_size_t j = 0; j < binding_count; ++j) {
      hipDeviceptr_t ptr = command_buffer->descriptor_sets[i].bindings[j];
      iree_host_size_t sz = command_buffer->descriptor_sets[i].binding_sizes[j];
      if (flags[j] == IREE_HAL_DESCRIPTOR_FLAG_NONE) {
        IREE_ASSERT(
            iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_write(
                command_buffer, ptr, sz, node)));
      } else {
        IREE_ASSERT(
            iree_status_is_ok(iree_hal_hip_stream_command_buffer_register_read(
                command_buffer, ptr, sz, node)));
      }
    }
  }
  if (!num_parents) {
    iree_hal_hip_stream_command_buffer_add_root_node(command_buffer, node);
  }
  command_buffer->total_elements++;
#endif
  // IREE_HIP_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
  //                                &command_buffer->tracing_event_list,
  //                                command_buffer->hip_stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need hip implementation of dispatch indirect");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_stream_command_buffer_vtable = {
        .destroy = iree_hal_hip_stream_command_buffer_destroy,
        .begin = iree_hal_hip_stream_command_buffer_begin,
        .end = iree_hal_hip_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_hip_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hip_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hip_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_hip_stream_command_buffer_signal_event,
        .reset_event = iree_hal_hip_stream_command_buffer_reset_event,
        .wait_events = iree_hal_hip_stream_command_buffer_wait_events,
        .discard_buffer = iree_hal_hip_stream_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_hip_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hip_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hip_stream_command_buffer_copy_buffer,
        .collective = iree_hal_hip_stream_command_buffer_collective,
        .push_constants = iree_hal_hip_stream_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_hip_stream_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_hip_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_hip_stream_command_buffer_dispatch_indirect,
};
