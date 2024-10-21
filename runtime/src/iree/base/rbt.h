// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_RBT_H
#define IREE_BASE_RBT_H

#include "iree/base/allocator.h"
#include "iree/base/config.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_tree_node_t iree_tree_node_t;
typedef struct iree_tree_t iree_tree_t;

typedef enum iree_tree_walk_type_e {
  IREE_TREE_WALK_PREORDER,
  IREE_TREE_WALK_INORDER,
  IREE_TREE_WALK_POSTORDER,
} iree_tree_walk_type_e;

typedef struct iree_tree_node_t iree_tree_node_t;
typedef struct iree_tree_node_t {
  bool red;
  iree_tree_node_t* left;
  iree_tree_node_t* right;
  iree_tree_node_t* parent;
  iree_host_size_t key;
  bool is_sentinel;
  uint8_t* data;
} iree_tree_node_t;

typedef struct iree_tree_t {
  uint32_t element_size;
  iree_allocator_t allocator;
  iree_tree_node_t* root;
  iree_host_size_t size;
  iree_tree_node_t* cache;  // Cache for deleted nodes
  iree_tree_node_t nil;
} iree_tree_t;

void* iree_tree_node_get_data(iree_tree_node_t* node);
iree_host_size_t iree_tree_node_get_key(iree_tree_node_t* node);

iree_status_t iree_tree_initialize(iree_allocator_t allocator,
                                   iree_host_size_t element_size,
                                   iree_tree_t* tree);
uint32_t iree_tree_get_data_size(iree_tree_t* tree);
void iree_tree_deinitialize(iree_tree_t* tree);
iree_status_t iree_tree_insert(iree_tree_t* tree, iree_host_size_t key,
                               iree_tree_node_t** out_data);
iree_host_size_t iree_tree_size(iree_tree_t* tree);
iree_status_t iree_tree_move_node(iree_tree_t* tree, iree_tree_node_t* node,
                                  iree_host_size_t new_key);
iree_tree_node_t* iree_tree_get(iree_tree_t* tree, iree_host_size_t key);
void iree_tree_walk(iree_tree_t* tree, iree_tree_walk_type_e walk_type,
                    bool (*callback)(iree_tree_node_t*, void*),
                    void* user_data);
iree_tree_node_t* iree_tree_node_next(iree_tree_node_t* node);
iree_tree_node_t* iree_tree_node_prev(iree_tree_node_t* node);
iree_tree_node_t* iree_tree_lower_bound(iree_tree_t* tree,
                                        iree_host_size_t key);
iree_tree_node_t* iree_tree_upper_bound(iree_tree_t* tree,
                                        iree_host_size_t key);
iree_tree_node_t* iree_tree_first(iree_tree_t* tree);
iree_tree_node_t* iree_tree_last(iree_tree_t* tree);
void iree_tree_erase(iree_tree_t* tree, iree_tree_node_t* node);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_RBT_H