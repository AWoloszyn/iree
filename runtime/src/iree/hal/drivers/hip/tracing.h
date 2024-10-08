// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_TRACING_H_
#define IREE_HAL_DRIVERS_HIP_TRACING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Per-stream HIP tracing context.
// No-op if IREE tracing is not enabled.
//
// Use the IREE_HIP_TRACE_* macros to trace a contiguous set of stream
// operations. Unlike the normal tracy macros there are no zone IDs and instead
// each stream gets an ID allocated once and passed to all tracing macros.
//
// Usage:
//   IREE_HIP_STREAM_TRACE_ZONE_BEGIN(queue->tracing_context, stream);
//   hipModuleLaunchKernel(..., stream);
//   IREE_HIP_STREAM_TRACE_ZONE_END(queue->tracing_context, stream);
//   ...
//   iree_hal_hip_tracing_context_collect(queue->tracing_context);
//
// NOTE: timestamps can have non-trivial side-effecting behavior and may
// introduce serialization in graph execution.
//
// TODO(benvanik): expose hipEvent_t reservation separate from recording. For
// graphs we will need to insert the events but in order to reuse the graphs
// we'll need to reserve and patch new events each graph launch.
//
// Thread-compatible: external synchronization is required if using from
// multiple threads (same as with hipStream_t itself).
typedef struct iree_hal_hip_tracing_context_t iree_hal_hip_tracing_context_t;
typedef struct iree_hal_hip_tracing_context_event_t
    iree_hal_hip_tracing_context_event_t;

// This is used when tracing is enabled. Calls to dispatch and event related
// functions will update the pointers to keep the list up to date.
typedef struct iree_hal_hip_tracing_context_event_list_t {
  iree_hal_hip_tracing_context_event_t* head;
  iree_hal_hip_tracing_context_event_t* tail;
} iree_hal_hip_tracing_context_event_list_t;

typedef enum iree_hal_hip_tracing_verbosity_e {
  IREE_HAL_HIP_TRACING_VERBOSITY_OFF = 0,
  IREE_HAL_HIP_TRACING_VERBOSITY_COARSE,
  IREE_HAL_HIP_TRACING_VERBOSITY_FINE,
  IREE_HAL_HIP_TRACING_VERBOSITY_MAX
} iree_hal_hip_tracing_verbosity_t;

// Allocates a tracing context for the given HIP |stream|.
// Each context must only be used with the stream it was created for.
iree_status_t iree_hal_hip_tracing_context_allocate(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_string_view_t queue_name, hipStream_t stream,
    iree_hal_hip_tracing_verbosity_t stream_tracing_verbosity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hip_tracing_context_t** out_context);

// Frees a tracing context and all associated HIP resources.
// All submissions using the resources must be completed prior to calling.
void iree_hal_hip_tracing_context_free(iree_hal_hip_tracing_context_t* context);

// Collects in-flight timestamp queries from the stream and feeds them to tracy.
// Must be called frequently (every submission, etc) to drain the backlog;
// tracing may start failing if the internal ringbuffer is exceeded.
void iree_hal_hip_tracing_context_collect(
    iree_hal_hip_tracing_context_t* context);

// Notifies that the given list of events has been dispached on to the gpu.
void iree_hal_hip_tracing_notify_submitted(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list);

// Frees the events and returns them back into the tracing context.
void iree_hal_hip_tracing_free(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

// Begins a normal zone derived on the calling |src_loc|.
// Must be perfectly nested and paired with a corresponding zone end.
void iree_hal_hip_stream_tracing_zone_begin_impl(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list, hipStream_t stream,
    iree_hal_hip_tracing_verbosity_t verbosity,
    const iree_tracing_location_t* src_loc);

// Begins an external zone using the given source information.
// The provided strings will be copied into the tracy buffer.
void iree_hal_hip_stream_tracing_zone_begin_external_impl(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list, hipStream_t stream,
    iree_hal_hip_tracing_verbosity_t verbosity, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length);

void iree_hal_hip_graph_tracing_zone_begin_external_impl(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list,
    hipGraphNode_t* out_node, hipGraph_t graph,
    iree_hal_hip_tracing_verbosity_t verbosity,
    hipGraphNode_t* dependency_nodes, size_t dependency_nodes_count,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);

void iree_hal_hip_stream_tracing_zone_end_impl(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list, hipStream_t stream,
    iree_hal_hip_tracing_verbosity_t verbosity);
void iree_hal_hip_graph_tracing_zone_end_impl(
    iree_hal_hip_tracing_context_t* context,
    iree_hal_hip_tracing_context_event_list_t* event_list,
    hipGraphNode_t* out_node, hipGraph_t graph,
    iree_hal_hip_tracing_verbosity_t verbosity,
    hipGraphNode_t* dependency_nodes, size_t dependency_nodes_count);

// Begins a new zone with the parent function name.
#define IREE_HIP_STREAM_TRACE_ZONE_BEGIN(context, event_list, stream,     \
                                         verbosity)                       \
  static const iree_tracing_location_t TracyConcat(                       \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__, \
                                            (uint32_t)__LINE__, 0};       \
  iree_hal_hip_stream_tracing_zone_begin_impl(                            \
      context, event_list, stream, verbosity,                             \
      &TracyConcat(__tracy_source_location, __LINE__));

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_HIP_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(                             \
    context, event_list, stream, verbosity, file_name, file_name_length, line, \
    function_name, function_name_length, name, name_length)                    \
  iree_hal_hip_stream_tracing_zone_begin_external_impl(                        \
      context, event_list, stream, verbosity, file_name, file_name_length,     \
      line, function_name, function_name_length, name, name_length)

#define IREE_HIP_GRAPH_TRACE_ZONE_BEGIN_EXTERNAL(                             \
    context, event_list, out_node, graph, verbosity, dependency_nodes,        \
    dependency_nodes_count, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                                  \
  iree_hal_hip_graph_tracing_zone_begin_external_impl(                        \
      context, event_list, out_node, graph, verbosity, dependency_nodes,      \
      dependency_nodes_count, file_name, file_name_length, line,              \
      function_name, function_name_length, name, name_length)

#define IREE_HIP_STREAM_TRACE_ZONE_END(context, event_list, stream, verbosity) \
  iree_hal_hip_stream_tracing_zone_end_impl(context, event_list, stream,       \
                                            verbosity)

#define IREE_HIP_GRAPH_TRACE_ZONE_END(context, event_list, out_node, graph,    \
                                      verbosity, dependency_nodes,             \
                                      dependency_nodes_count)                  \
  iree_hal_hip_graph_tracing_zone_end_impl(context, event_list, out_node,      \
                                           graph, verbosity, dependency_nodes, \
                                           dependency_nodes_count)
#else

#define IREE_HIP_STREAM_TRACE_ZONE_BEGIN(context, event_list, stream, verbosity)
#define IREE_HIP_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(                             \
    context, event_list, stream, verbosity, file_name, file_name_length, line, \
    function_name, function_name_length, name, name_length)
#define IREE_HIP_GRAPH_TRACE_ZONE_BEGIN_EXTERNAL(                             \
    context, event_list, out_node, graph, verbosity, dependency_nodes,        \
    dependency_nodes_count, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)
#define IREE_HIP_STREAM_TRACE_ZONE_END(context, evnet_list, stream, verbosity)
#define IREE_HIP_GRAPH_TRACE_ZONE_END(context, event_list, out_node, graph, \
                                      verbosity, dependency_nodes,          \
                                      dependency_nodes_count)
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_TRACING_H_
