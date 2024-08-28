// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_GROUP_DEVICE_H_
#define IREE_HAL_UTILS_GROUP_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_group_device_interface_t iree_hal_group_device_interface_t;

iree_status_t iree_hal_utils_group_device_create(
    iree_host_size_t device_count,
    iree_hal_device_t** devices,
    bool supports_memory_pools,
    iree_hal_group_device_interface_t* interface,
    iree_hal_device_t** out_device
);


typedef struct iree_hal_group_device_interface_vtable_t iree_hal_group_device_interface_vtable_t;
typedef struct iree_hal_group_device_interface_t {
    iree_hal_group_device_interface_vtable_t* vtable;
} iree_hal_group_device_interface_t;

typedef struct iree_hal_group_device_interface_vtable_t {
    iree_status_t (*create_executable_cache)(
      iree_hal_group_device_interface_vtable_t* interface, iree_string_view_t identifier,
      iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache);

} iree_hal_group_device_interface_vtable_t;

#ifdef __cplusplus
}
#endif  // __cplusplus


#endif // IREE_HAL_UTILS_GROUP_DEVICE_H_
