// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_queue.h"

#include "iree/hal/drivers/hip/status_util.h"

#define IREE_HIP_QUEUE_MAX_STREAMS 32

typedef struct iree_hal_hip_queue_t {
  const iree_hal_hip_dynamic_symbols_t* symbols;
  iree_allocator_t host_allocator;
  hipStream_t streams[IREE_HIP_QUEUE_MAX_STREAMS * 2];
} iree_hal_hip_queue_t;

iree_status_t iree_hal_hip_queue_create(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_allocator_t host_allocator, iree_hal_hip_queue_t** out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_queue_t* queue;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(iree_hal_hip_queue_t), (void**)&queue));

  for (uint32_t i = 0; i < IREE_HIP_QUEUE_MAX_STREAMS; ++i) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols,
        hipStreamCreateWithFlags(&queue->streams[i], hipStreamNonBlocking),
        "hipStreamCreateWithFlags");
  }

  queue->symbols = symbols;
  IREE_TRACE_ZONE_END(z0);
  out_queue[0] = queue;
  return iree_ok_status();
}

hipStream_t iree_hal_hip_queue_get_stream(iree_hal_hip_queue_t* queue,
                                          int32_t stream_idx) {
  IREE_ASSERT(stream_idx < IREE_HIP_QUEUE_MAX_STREAMS);
  return queue->streams[stream_idx];
}

void iree_hal_hip_queue_destroy(iree_hal_hip_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (uint32_t i = 0; i < IREE_HIP_QUEUE_MAX_STREAMS; ++i) {
    // IREE_HIP_IGNORE_ERROR(queue->symbols,
    // hipStreamDestroy(queue->streams[i]));
  }

  iree_allocator_free(queue->host_allocator, queue);

  IREE_TRACE_ZONE_END(z0);
}
