// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_HASH_MAP_H
#define IREE_BASE_HASH_MAP_H

#include "iree/base/allocator.h"
#include "iree/base/config.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hash_map_element_t iree_hash_map_element_t;
typedef struct iree_hash_map_t iree_hash_map_t;

void* iree_hash_map_element_get_data(iree_hash_map_element_t* element);
iree_host_size_t iree_hash_map_element_get_key(
    iree_hash_map_element_t* element);

iree_status_t iree_hash_map_create(iree_allocator_t allocator,
                                   iree_host_size_t element_size,
                                   iree_host_size_t num_builtin_elements,
                                   iree_hash_map_t** out);
void iree_hash_map_destroy(iree_hash_map_t* map);
void iree_hash_map_walk(iree_hash_map_t* hash_map,
                        bool (*callback)(iree_hash_map_element_t*, void*),
                        void* user_data);

iree_status_t iree_hash_map_insert(iree_hash_map_t* map, iree_host_size_t key,
                                   iree_hash_map_element_t** element);
iree_status_t iree_hash_map_try_insert(iree_hash_map_t* map,
                                       iree_host_size_t key,
                                       iree_hash_map_element_t** element);
void iree_hash_map_erase(iree_hash_map_t* map,
                         iree_hash_map_element_t* element);
iree_hash_map_element_t* iree_hash_map_find(iree_hash_map_t* map,
                                            iree_host_size_t key);
iree_host_size_t iree_hash_map_size(iree_hash_map_t* map);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_HASH_MAP_H
