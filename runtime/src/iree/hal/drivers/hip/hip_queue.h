// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_QUEUE_H_
#define IREE_HAL_DRIVERS_HIP_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_hip_queue_t iree_hal_hip_queue_t;

// Creates a device that owns and manages its own hipCtx_t.
iree_status_t iree_hal_hip_queue_create(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_allocator_t host_allocator, iree_hal_hip_queue_t** out_device);

hipStream_t iree_hal_hip_queue_get_stream(iree_hal_hip_queue_t* queue,
                                          int32_t stream_idx);

void iree_hal_hip_queue_destroy(iree_hal_hip_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_DEVICE_H_
