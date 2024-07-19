// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_CIRCULAR_ARRAY_H
#define IREE_BASE_CIRCULAR_ARRAY_H

#include "iree/base/allocator.h"
#include "iree/base/config.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_circular_array_t iree_circular_array_t;
iree_status_t iree_circular_array_create(iree_allocator_t host_allocator,
                                         iree_host_size_t capacity,
                                         iree_host_size_t element_size,
                                         iree_circular_array_t** array);
void iree_circular_array_free(iree_circular_array_t* array);

iree_status_t iree_circular_array_push_back(iree_circular_array_t* array,
                                            void** out_item);
void iree_circular_array_front(iree_circular_array_t* array, void** out_item);
void iree_circular_array_pop_front(iree_circular_array_t* array);
iree_host_size_t iree_circular_array_size(iree_circular_array_t* array);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // IREE_BASE_CIRCULAR_ARRAY_H
