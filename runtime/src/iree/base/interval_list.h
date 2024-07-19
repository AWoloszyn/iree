// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERVAL_LIST_H
#define IREE_BASE_INTERVAL_LIST_H

#include "iree/base/allocator.h"
#include "iree/base/config.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_interval_list is a list of non-overlapping intervals.
// They can be contiguous or otherwise, but they never overlap.
// Inserting intervals will over-write any intervals that
// use the same range.

typedef struct iree_tree_node_t iree_interval_t;
typedef struct iree_tree_t iree_interval_list_t;

iree_interval_t* iree_interval_next(iree_interval_t* interval);
iree_interval_t* iree_interval_previous(iree_interval_t* interval);
void* iree_interval_get_data(iree_interval_t* interval);
uint64_t iree_interval_base(iree_interval_t* interval);
uint64_t iree_interval_size(iree_interval_t* interval);

iree_status_t iree_interval_list_create(iree_allocator_t allocator,
                                        iree_host_size_t element_size,
                                        iree_interval_list_t** out);
void iree_interval_list_free(iree_interval_list_t* list);
void iree_interval_list_print(iree_interval_list_t* list,
                              void (*print_element)(FILE*, void*, void*),
                              void* user_data);

// Inserts an a range into the interval list, and over-writes anything that
// might be in that location.
iree_status_t iree_interval_list_insert(iree_interval_list_t* interval_list,
                                        uint64_t offset, uint64_t size,
                                        iree_interval_t** out);

// Inserts some number of elements into the interval list, such that
// any existing interval remains, but any gaps are filled by new
// intervals. Returns all nodes that overlap the range.
iree_status_t iree_interval_list_insert_no_overwrite(
    iree_interval_list_t* interval_list, uint64_t offset, uint64_t size,
    iree_interval_t** out_begin, iree_interval_t** out_end);
// begin and end work like C++ iterators, end is one past the last element
// (don't dereference it or use it)
void iree_interval_list_find(iree_interval_list_t* interval_list,
                             uint64_t offset, uint64_t size,
                             iree_interval_t** begin, iree_interval_t** end);
iree_status_t iree_interval_list_erase(iree_interval_list_t* interval_list,
                                       uint64_t offset, uint64_t size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERVAL_LIST_H
