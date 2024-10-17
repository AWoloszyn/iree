// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INDEXABLE_DEQUE_H__
#define IREE_BASE_INDEXABLE_DEQUE_H__
#include "iree/base/alignment.h"
#include "iree/base/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

// The indexable deque essentially a circular array
// where we can push to the back and pop from the front.
// The helper functions allow you to index into the array.
// Furthermore an initial allocation may be provided inline
// as an optimization.
typedef struct iree_indexable_queue_t {
  iree_allocator_t allocator;
  uint8_t* elements;
  iree_host_size_t element_size;
  iree_host_size_t element_count;
  iree_host_size_t capacity;
  iree_host_size_t head;
  iree_alignas(iree_max_align_t) uint8_t initial_allocation[];
} iree_indexable_queue_t;

// Initializes the indexable_queue with the given allocator, element size and
// number of inline elements (which may be 0).
void iree_indexable_queue_initialize(iree_indexable_queue_t* queue,
                                     iree_allocator_t allocator,
                                     iree_host_size_t element_size,
                                     iree_host_size_t inline_count);

// Deinitializes the list, it does not have to be empty.
iree_status_t iree_indexable_queue_deinitilaize(iree_indexable_queue_t* queue);

// Copies the given element into the back of the array. This may cause a
// re-allocation of data.
iree_status_t iree_indexable_queue_push_back(iree_indexable_queue_t* queue,
                                             void* element);

// Pops the element from the front of the array and moves the head.
void iree_indexable_queue_pop_front(iree_indexable_queue_t* queue,
                                    iree_host_size_t count);

// Returns a pointer to the element at index i
void* iree_indexable_queue_at(iree_indexable_queue_t* queue,
                              iree_host_size_t i);

#define IREE_TYPED_INDEXABLE_QUEUE_WRAPPER(name, type, default_element_count)  \
  typedef struct name##_indexable_queue_t {                                    \
    iree_allocator_t allocator;                                                \
    void* elements;                                                            \
    iree_host_size_t element_size;                                             \
    iree_host_size_t element_count;                                            \
    iree_host_size_t capacity;                                                 \
    iree_host_size_t head;                                                     \
    iree_alignas(iree_max_align_t) uint8_t                                     \
        initial_allocation[default_element_count * sizeof(type)];              \
  } name##_indexable_queue_t;                                                  \
  static inline void name##_indexable_queue_initialize(                        \
      name##_indexable_queue_t* out_queue, iree_allocator_t allocator) {       \
    iree_indexable_queue_initialize((iree_indexable_queue_t*)out_queue,        \
                                    allocator, sizeof(type), default_size);    \
  }                                                                            \
  static inline void name##_indexable_queue_deinitialize(                      \
      name##_indexable_queue_t* out_queue, iree_allocator_t allocator) {       \
    iree_indexable_queue_deinitialize((iree_indexable_queue_t*)out_queue);     \
  }                                                                            \
  iree_status_t name##_indexable_queue_push_back(                              \
      name##_indexable_queue_t* queue, type element) {                         \
    return iree_indexable_queue_push_back((iree_indexable_queue_t*)queue,      \
                                          &element);                           \
  }                                                                            \
  void name##_indexable_queue_pop_front(name##_indexable_queue_t* queue,       \
                                        iree_host_size_t count) {              \
    return iree_indexable_queue_pop_front((iree_indexable_queue_t*)queue,      \
                                          count);                              \
  }                                                                            \
  type name##_indexable_queue_at(name##_indexable_queue_t* queue,              \
                                 iree_host_size_t i) {                         \
    type t;                                                                    \
    memcpy(&t, iree_indexable_queue_at((iree_indexable_queue_t*)queue, count), \
           sizeof(type));                                                      \
    return t;                                                                  \
  }

#ifdef __cplusplus
}
#endif

#endif  //  IREE_BASE_INTERNAL_INDEXABLE_DEQUE_H__
