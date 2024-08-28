// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_GROUP_COMMAND_BUFFER_H_
#define IREE_HAL_UTILS_GROUP_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_utils_group_command_buffer_create(
    iree_allocator_t host_allocator,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t** command_buffers
);

#ifdef __cplusplus
}
#endif  // __cplusplus


#endif // IREE_HAL_UTILS_GROUP_COMMAND_BUFFER_H_
