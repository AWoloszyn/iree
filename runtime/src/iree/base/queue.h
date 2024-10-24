// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_QUEUE_H__
#define IREE_BASE_QUEUE_H__

#include "iree/base/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

// The queue essentially a circular array
// where we can push to the back and pop from the front.
// The helper functions allow you to index into the array.
// Furthermore an initial allocation may be provided inline
// as an optimization.
typedef struct iree_queue_t {
  iree_allocator_t allocator;
  uint8_t* elements;
  iree_host_size_t element_size;
  iree_host_size_t element_count;
  iree_host_size_t capacity;
  iree_host_size_t head;
  uint8_t initial_allocation[];
} iree_queue_t;

// Initializes the queue with the given allocator, element size and
// number of inline elements (which may be 0).
void iree_queue_initialize(iree_queue_t* queue, iree_allocator_t allocator,
                           iree_host_size_t element_size,
                           iree_host_size_t inline_count);

// Deinitializes the list, it does not have to be empty.
void iree_queue_deinitialize(iree_queue_t* queue);

// Copies the given element into the back of the array. This may cause a
// re-allocation of data.
iree_status_t iree_queue_push_back(iree_queue_t* queue, void* element);

// Pops the element from the front of the array and moves the head.
void iree_queue_pop_front(iree_queue_t* queue, iree_host_size_t count);

// Returns a pointer to the element at index i
void* iree_queue_at(iree_queue_t* queue, iree_host_size_t i);

#define IREE_TYPED_QUEUE_WRAPPER(name, type, default_element_count)          \
  typedef struct name##_t {                                                  \
    iree_allocator_t allocator;                                              \
    void* elements;                                                          \
    iree_host_size_t element_size;                                           \
    iree_host_size_t element_count;                                          \
    iree_host_size_t capacity;                                               \
    iree_host_size_t head;                                                   \
    iree_alignas(iree_max_align_t) uint8_t                                   \
        initial_allocation[default_element_count * sizeof(type)];            \
  } name##_t;                                                                \
  static inline void name##_initialize(name##_t* out_queue,                  \
                                       iree_allocator_t allocator) {         \
    iree_queue_initialize((iree_queue_t*)out_queue, allocator, sizeof(type), \
                          default_element_count);                            \
  }                                                                          \
  static inline void name##_deinitialize(name##_t* out_queue) {              \
    iree_queue_deinitialize((iree_queue_t*)out_queue);                       \
  }                                                                          \
  iree_status_t name##_push_back(name##_t* queue, type element) {            \
    return iree_queue_push_back((iree_queue_t*)queue, &element);             \
  }                                                                          \
  void name##_pop_front(name##_t* queue, iree_host_size_t count) {           \
    return iree_queue_pop_front((iree_queue_t*)queue, count);                \
  }                                                                          \
  type name##_at(name##_t* queue, iree_host_size_t i) {                      \
    type t;                                                                  \
    memcpy(&t, iree_queue_at((iree_queue_t*)queue, i), sizeof(type));        \
    return t;                                                                \
  }

#ifdef __cplusplus
}
#endif

#endif  //  IREE_BASE_INTERNAL_QUEUE_H__