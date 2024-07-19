// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/base/hash_map.h"
#include "iree/base/interval_list.h"
#include "iree/base/rbt.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct element_type {
  size_t val;
};
// Tests general parser usage.
TEST(HashmapTest, basic) {
  srand(time(NULL));
  std::vector<size_t> values;
  for (size_t i = 0; i < 170; ++i) {
    values.push_back(rand());
  }

  iree_hash_map_t* hm;
  IREE_ASSERT_OK(iree_hash_map_create(iree_allocator_system(),
                                      sizeof(element_type), 8, &hm));
  for (auto& i : values) {
    iree_hash_map_element_t* element;
    IREE_ASSERT_OK(iree_hash_map_insert(hm, i, &element));
    ((element_type*)iree_hash_map_element_get_data(element))->val = ~i;
  }

  for (auto& i : values) {
    iree_hash_map_element_t* element = iree_hash_map_find(hm, i);
    EXPECT_NE(element, nullptr);
    EXPECT_EQ(((element_type*)iree_hash_map_element_get_data(element))->val,
              ~i);
  }

  for (size_t i = 0; i < values.size(); ++i) {
    if ((i % 2) == 0) {
      iree_hash_map_element_t* element = iree_hash_map_find(hm, values[i]);
      EXPECT_NE(element, nullptr);
      iree_hash_map_erase(hm, element);
    }
  }
  EXPECT_EQ(values.size() / 2, iree_hash_map_size(hm));

  struct st {
    size_t n = 0;
    std::vector<size_t>* vals;
  } st_val;
  st_val.vals = &values;
  iree_hash_map_walk(
      hm,
      +[](iree_hash_map_element_t* element, void* n_) -> bool {
        st* n = (st*)n_;
        n->n++;
        auto it = std::find(n->vals->begin(), n->vals->end(),
                            iree_hash_map_element_get_key(element));
        EXPECT_NE(it, n->vals->end());
        EXPECT_TRUE((it - n->vals->begin()) % 2 == 1);
        return true;
      },
      &st_val);
  EXPECT_EQ(st_val.n, values.size() / 2);
}

// Tests general parser usage.
TEST(MapTest, basic) {
  time_t x = time(NULL);
  fprintf(stderr, "TIME %ld\n", x);

  srand(x);
  std::vector<size_t> values;
  for (size_t i = 0; i < 170; ++i) {
    values.push_back(rand());
  }

  iree_tree_t* tree;
  IREE_ASSERT_OK(
      iree_tree_create(iree_allocator_system(), sizeof(element_type), &tree));
  for (auto& i : values) {
    iree_tree_node_t* element;
    IREE_ASSERT_OK(iree_tree_insert(tree, i, &element));
    ((element_type*)iree_tree_node_get_data(element))->val = ~i;
  }

  for (auto& i : values) {
    iree_tree_node_t* element = iree_tree_get(tree, i);
    EXPECT_NE(element, nullptr);
    EXPECT_EQ(((element_type*)iree_tree_node_get_data(element))->val, ~i);
  }

  for (size_t i = 0; i < values.size(); ++i) {
    if ((i % 2) == 0) {
      iree_tree_node_t* element = iree_tree_get(tree, values[i]);
      EXPECT_NE(element, nullptr);
      iree_tree_erase(tree, element);
    }
  }
  EXPECT_EQ(values.size() / 2, iree_tree_size(tree));

  struct st {
    size_t n = 0;
    std::vector<size_t>* vals;
  } st_val;
  st_val.vals = &values;
  iree_tree_walk(
      tree,
      +[](iree_tree_node_t* element, void* n_) -> bool {
        st* n = (st*)n_;
        n->n++;
        auto it = std::find(n->vals->begin(), n->vals->end(),
                            iree_tree_node_get_key(element));
        EXPECT_NE(it, n->vals->end());
        EXPECT_TRUE((it - n->vals->begin()) % 2 == 1);
        return true;
      },
      &st_val);
  EXPECT_EQ(st_val.n, values.size() / 2);

  iree_tree_node_t* first = iree_tree_first(tree);
  iree_tree_node_t* prev_first = NULL;
  iree_host_size_t last_key = 0;
  do {
    iree_host_size_t new_key = iree_tree_node_get_key(first);
    EXPECT_GE(new_key, last_key);
    last_key = new_key;
    prev_first = first;
    first = iree_tree_node_next(first);
  } while (first);

  iree_tree_node_t* last = iree_tree_last(tree);
  EXPECT_EQ(last, prev_first);
}

// Tests general parser usage.
TEST(IntervalListTest, basic) {
  time_t x = time(NULL);
  fprintf(stderr, "TIME %ld\n", x);

  srand(x);
  std::vector<size_t> values;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < 10000; ++i) {
    values.push_back(rand() % (10000 - 1000));
    sizes.push_back(rand() % 999 + 1);
  }

  iree_interval_list_t* list;
  IREE_ASSERT_OK(iree_interval_list_create(iree_allocator_system(),
                                           sizeof(element_type), &list));

  for (size_t i = 0; i < values.size(); ++i) {
    iree_interval_t* begin;
    iree_interval_t* end;
    IREE_ASSERT_OK(iree_interval_list_insert_no_overwrite(
        list, values[i], sizes[i], &begin, &end));
    while (begin != end) {
      ((element_type*)iree_interval_get_data(begin))->val += 1;
      begin = iree_interval_next(begin);
    }
  }

  std::vector<std::pair<iree_host_size_t, iree_host_size_t>> random_samples;
  for (size_t i = 0; i < 100; ++i) {
    size_t rand_num = rand() % 100000;
    size_t ct = 0;
    for (size_t j = 0; j < values.size(); ++j) {
      ct += (rand_num >= values[j] && rand_num < values[j] + sizes[j]);
    }
    iree_interval_t* begin;
    iree_interval_t* end;
    iree_interval_list_find(list, rand_num, 1, &begin, &end);
    ASSERT_TRUE(begin != end || ct == 0);
    if (ct != 0) {
      EXPECT_EQ(ct, ((element_type*)iree_interval_get_data(begin))->val);
      ASSERT_EQ(end, iree_interval_next(begin));
    }
  }

  iree_interval_list_free(list);
}

TEST(CircularArrayTest, basic) {
  for (size_t i = 0; i < 100; ++i) {
  }
}

}  // namespace
}  // namespace iree
