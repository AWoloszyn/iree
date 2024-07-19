// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/hash_map.h"

#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/config.h"
#include "stdint.h"

#define IREE_HASH_MAP_MIN_ELEMENTS 32
#define IREE_MAX_MAP_CAPACITY 0.8

typedef enum hash_map_element_state {
  EMPTY = 0,
  FULL = 1,
  USED = 2,

} hash_map_element_state;

struct iree_hash_map_element_t {
  hash_map_element_state state;  // for alignment reasons
  iree_host_size_t key;
  uint8_t data[];
};

struct iree_hash_map_t {
  uint32_t element_size;
  iree_allocator_t allocator;
  iree_host_size_t element_count;
  iree_host_size_t capacity;
  iree_hash_map_element_t* elements;
  iree_hash_map_element_t initial_elements[];
};

const uint64_t k_valid_hash_bits = 0xFFFFFFFFFFF;
static uint64_t next_hash(uint64_t hash) {
  return ((hash * 69069) + 1) & k_valid_hash_bits;
}
iree_host_size_t iree_hash_map_size(iree_hash_map_t* map) {
  return map->element_count;
}
iree_hash_map_element_t* iree_hash_map_get_element_by_index(
    iree_hash_map_element_t* elements, uint32_t element_size,
    iree_host_size_t index) {
  return (iree_hash_map_element_t*)((uint8_t*)(elements) +
                                    index * (sizeof(iree_hash_map_element_t) +
                                             ((element_size + 7) & ~7)));
}

void* iree_hash_map_element_get_data(iree_hash_map_element_t* element) {
  IREE_ASSERT(element);
  return element->data;
}

iree_host_size_t iree_hash_map_element_get_key(
    iree_hash_map_element_t* element) {
  IREE_ASSERT(element);
  return element->key;
}

iree_status_t iree_hash_map_create(iree_allocator_t allocator,
                                   iree_host_size_t element_size,
                                   iree_host_size_t num_builtin_elements,
                                   iree_hash_map_t** out) {
  iree_hash_map_t* t;
  iree_status_t status = iree_allocator_malloc(
      allocator,
      sizeof(iree_hash_map_t) +
          num_builtin_elements *
              (sizeof(iree_hash_map_element_t) + ((element_size + 7) & ~7)),
      (void**)&t);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    return status;
  }
  t->element_size = element_size;
  t->allocator = allocator;
  t->capacity = num_builtin_elements;
  t->elements = t->initial_elements;
  *out = t;
  return status;
}

void iree_hash_map_walk(iree_hash_map_t* hash_map,
                        bool (*callback)(iree_hash_map_element_t*, void*),
                        void* user_data) {
  for (iree_host_size_t i = 0; i < hash_map->capacity; ++i) {
    iree_hash_map_element_t* element = iree_hash_map_get_element_by_index(
        hash_map->elements, hash_map->element_size, i);
    if (element->state == FULL) {
      if (!callback(element, user_data)) {
        return;
      }
    }
  }
}

void iree_hash_map_destroy(iree_hash_map_t* map) {
  if (map->elements != map->initial_elements) {
    iree_allocator_free(map->allocator, map->elements);
  }
  iree_allocator_free(map->allocator, map);
}

iree_status_t iree_hash_map_insert_internal(iree_hash_map_t* map,
                                            iree_host_size_t key,
                                            iree_hash_map_element_t** element,
                                            bool ignore_if_already_present) {
  if ((float)map->element_count / (float)map->capacity >
          IREE_MAX_MAP_CAPACITY ||
      map->capacity < 8) {
    // Time for a resize!
    iree_hash_map_element_t* old_elements = map->elements;
    iree_status_t status = iree_allocator_malloc(
        map->allocator,
        map->capacity * 2 *
            (sizeof(iree_hash_map_element_t) + ((map->element_size + 7) & ~7)),
        (void**)&map->elements);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      map->elements = old_elements;
      return status;
    }
    iree_host_size_t old_capacity = map->capacity;
    map->capacity *= 2;
    map->element_count = 0;

    for (iree_host_size_t i = 0; i < old_capacity; ++i) {
      iree_hash_map_element_t* old_element = iree_hash_map_get_element_by_index(
          old_elements, map->element_size, i);
      if (old_element->state == FULL) {
        iree_hash_map_element_t* inserted;
        // The only failure case is allocator fail, but we know for a fact this
        // wont cause a resize.
        IREE_ASSERT(iree_status_is_ok(
            iree_hash_map_insert(map, old_element->key, &inserted)));
        memcpy(inserted->data, old_element->data, map->element_size);
      }
    }
    if (old_elements != map->initial_elements) {
      iree_allocator_free(map->allocator, old_elements);
    }
  }

  uint64_t hash = next_hash(key);
  iree_hash_map_element_t* insert_location = NULL;
  iree_hash_map_element_t* potential_location = NULL;
  for (iree_host_size_t i = 0; i < map->capacity; ++i, hash = next_hash(hash)) {
    iree_host_size_t lookup = hash & (map->capacity - 1);
    bool found = false;

    insert_location = iree_hash_map_get_element_by_index(
        map->elements, map->element_size, lookup);
    switch (insert_location->state) {
      case EMPTY:
        found = true;
        break;
      case USED:  // Used but not currently full
        if (potential_location == NULL) {
          potential_location = insert_location;
        }
        found = true;
        continue;
      case FULL:
        if (insert_location->key == key) {
          found = true;
          break;
        }
        continue;
    }
    if (found) {
      break;
    }
  }
  IREE_ASSERT(potential_location ||
              insert_location && "could not find somewhere to put this");
  if (insert_location->state == FULL) {
    return ignore_if_already_present
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "Trying to insert a duplicate key");
  }
  insert_location = potential_location ? potential_location : insert_location;
  insert_location->key = key;
  insert_location->state = FULL;
  *element = insert_location;
  map->element_count++;
  return iree_ok_status();
}

iree_status_t iree_hash_map_insert(iree_hash_map_t* map, iree_host_size_t key,
                                   iree_hash_map_element_t** element) {
  return iree_hash_map_insert_internal(map, key, element, false);
}

iree_status_t iree_hash_map_try_insert(iree_hash_map_t* map,
                                       iree_host_size_t key,
                                       iree_hash_map_element_t** element) {
  return iree_hash_map_insert_internal(map, key, element, true);
}

void iree_hash_map_erase(iree_hash_map_t* map,
                         iree_hash_map_element_t* element) {
  element->state = USED;
  map->element_count--;
}

iree_hash_map_element_t* iree_hash_map_find(iree_hash_map_t* map,
                                            iree_host_size_t key) {
  uint64_t hash = next_hash(key);
  iree_hash_map_element_t* location = NULL;
  for (iree_host_size_t i = 0; i < map->capacity; ++i, hash = next_hash(hash)) {
    iree_host_size_t lookup = hash & (map->capacity - 1);

    location = iree_hash_map_get_element_by_index(map->elements,
                                                  map->element_size, lookup);
    switch (location->state) {
      case EMPTY:
        return NULL;
      case USED:  // Used but not currently full
        continue;
      case FULL:
        if (location->key == key) {
          return location;
        }
        continue;
    }
  }
  return NULL;
}
