// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "rbt.h"

#include "iree/testing/gtest.h"

class RedBlackTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_allocator_t allocator = iree_allocator_system();
    IREE_CHECK_OK(iree_tree_initialize(allocator, sizeof(int), initial_cache,
                                       1024, &tree_));
  }

  void TearDown() override { iree_tree_deinitialize(&tree_); }

  iree_tree_t tree_;
  uint8_t initial_cache[1024];
};

TEST_F(RedBlackTreeTest, initialize) { EXPECT_EQ(iree_tree_size(&tree_), 0); }

TEST_F(RedBlackTreeTest, insert) {
  iree_tree_node_t* node;
  EXPECT_EQ(iree_tree_insert(&tree_, 10, &node), iree_ok_status());
  EXPECT_EQ(iree_tree_size(&tree_), 1);
  EXPECT_EQ(iree_tree_node_get_key(node), 10);
}

TEST_F(RedBlackTreeTest, get) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  EXPECT_NE(iree_tree_get(&tree_, 10), nullptr);
  EXPECT_EQ(iree_tree_get(&tree_, 20), nullptr);
}

TEST_F(RedBlackTreeTest, delete) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_erase(&tree_, node);
  EXPECT_EQ(iree_tree_get(&tree_, 10), nullptr);
  EXPECT_EQ(iree_tree_size(&tree_), 0);
}

TEST_F(RedBlackTreeTest, walk) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  static_cast<int*>(iree_tree_node_get_data(node))[0] = 10;
  iree_tree_insert(&tree_, 20, &node);
  static_cast<int*>(iree_tree_node_get_data(node))[0] = 20;
  iree_tree_insert(&tree_, 30, &node);
  static_cast<int*>(iree_tree_node_get_data(node))[0] = 30;

  int sum = 0;
  auto callback = [](iree_tree_node_t* node, void* user_data) -> bool {
    int* sum = static_cast<int*>(user_data);
    EXPECT_EQ(*static_cast<int*>(iree_tree_node_get_data(node)),
              iree_tree_node_get_key(node));
    *sum += *static_cast<int*>(iree_tree_node_get_data(node));
    return true;
  };
  iree_tree_walk(&tree_, IREE_TREE_WALK_PREORDER, callback, &sum);
  EXPECT_EQ(sum, 60);
}

TEST_F(RedBlackTreeTest, boundary_conditions) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_insert(&tree_, 20, &node);
  iree_tree_insert(&tree_, 30, &node);

  EXPECT_EQ(iree_tree_node_get_key(iree_tree_first(&tree_)), 10);
  EXPECT_EQ(iree_tree_node_get_key(iree_tree_last(&tree_)), 30);
  EXPECT_EQ(iree_tree_node_get_key(iree_tree_lower_bound(&tree_, 15)), 20);
  EXPECT_EQ(iree_tree_node_get_key(iree_tree_upper_bound(&tree_, 15)), 20);
}

TEST_F(RedBlackTreeTest, move_node) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_move_node(&tree_, node, 20);
  EXPECT_EQ(iree_tree_get(&tree_, 10), nullptr);
  EXPECT_NE(iree_tree_get(&tree_, 20), nullptr);
}

TEST_F(RedBlackTreeTest, in_order_iterators) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_insert(&tree_, 20, &node);
  iree_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  for (iree_tree_node_t* node = iree_tree_first(&tree_); node != nullptr;
       node = iree_tree_node_next(node)) {
    keys.push_back(iree_tree_node_get_key(node));
  }

  EXPECT_EQ(keys.size(), 3);
  EXPECT_EQ(keys[0], 10);
  EXPECT_EQ(keys[1], 20);
  EXPECT_EQ(keys[2], 30);
}

TEST_F(RedBlackTreeTest, in_order_iterators_last) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_insert(&tree_, 20, &node);
  iree_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  for (iree_tree_node_t* node = iree_tree_last(&tree_); node != nullptr;
       node = iree_tree_node_prev(node)) {
    keys.push_back(iree_tree_node_get_key(node));
  }

  EXPECT_EQ(keys.size(), 3);
  EXPECT_EQ(keys[0], 30);
  EXPECT_EQ(keys[1], 20);
  EXPECT_EQ(keys[2], 10);
}

class RedBlackTreeWalkTest
    : public RedBlackTreeTest,
      public ::testing::WithParamInterface<iree_tree_walk_type_e> {};

TEST_P(RedBlackTreeWalkTest, walk) {
  iree_tree_node_t* node;
  iree_tree_insert(&tree_, 10, &node);
  iree_tree_insert(&tree_, 20, &node);
  iree_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  auto callback = [](iree_tree_node_t* node, void* user_data) -> bool {
    auto* keys = static_cast<std::vector<int>*>(user_data);
    keys->push_back(iree_tree_node_get_key(node));
    return true;
  };
  iree_tree_walk(&tree_, GetParam(), callback, &keys);

  if (GetParam() == IREE_TREE_WALK_INORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 10);
    EXPECT_EQ(keys[1], 20);
    EXPECT_EQ(keys[2], 30);
  } else if (GetParam() == IREE_TREE_WALK_PREORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 20);  // Assuming 20 is the root after balancing
    EXPECT_EQ(keys[1], 10);
    EXPECT_EQ(keys[2], 30);
  } else if (GetParam() == IREE_TREE_WALK_POSTORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 10);
    EXPECT_EQ(keys[1], 30);
    EXPECT_EQ(keys[2], 20);  // Assuming 20 is the root after balancing
  }
}

INSTANTIATE_TEST_SUITE_P(WalkTypes, RedBlackTreeWalkTest,
                         ::testing::Values(IREE_TREE_WALK_PREORDER,
                                           IREE_TREE_WALK_INORDER,
                                           IREE_TREE_WALK_POSTORDER));
