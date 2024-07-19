// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/interval_list.h"

#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/config.h"
#include "iree/base/rbt.h"
#include "iree/base/tracing.h"
#include "stdint.h"

typedef struct iree_tree_node_t iree_interval_t;
typedef struct iree_tree_t iree_interval_list_t;

typedef struct iree_interval_list_interval_data_t {
  uint64_t offset;
  uint64_t size;
  uint8_t data[];
} iree_interval_list_interval_data_t;

iree_interval_t* iree_interval_next(iree_interval_t* interval) {
  return iree_tree_node_next(interval);
}

iree_interval_t* iree_interval_previous(iree_interval_t* interval) {
  return iree_tree_node_prev(interval);
}

void* iree_interval_get_data(iree_interval_t* interval) {
  return ((iree_interval_list_interval_data_t*)iree_tree_node_get_data(
              interval))
      ->data;
}

uint64_t iree_interval_offset(iree_interval_t* interval) {
  return ((iree_interval_list_interval_data_t*)iree_tree_node_get_data(
              interval))
      ->offset;
}

uint64_t iree_interval_size(iree_interval_t* interval) {
  return ((iree_interval_list_interval_data_t*)iree_tree_node_get_data(
              interval))
      ->size;
}

iree_status_t iree_interval_list_create(iree_allocator_t allocator,
                                        iree_host_size_t element_size,
                                        iree_interval_list_t** out) {
  return iree_tree_create(
      allocator, sizeof(iree_interval_list_interval_data_t) + element_size,
      out);
}

void iree_interval_list_free(iree_interval_list_t* list) {
  return iree_tree_free(list);
}

typedef struct iree_interval_list_print_struct {
  iree_interval_t* last;
} iree_interval_list_print_struct;

void iree_interval_print(iree_interval_t* begin, iree_interval_t* end,
                         void (*print_element)(FILE*, void*, void*),
                         void* user_data, bool inclusive) {
  if (begin == end) {
    fprintf(stderr, "------\n");
    fprintf(stderr, "------\n");
    return;
  }
  bool run = false;
  uint64_t last_end = 0;
  if (inclusive) {
    if (end) end = iree_tree_node_next(end);
  }
  while (begin != end) {
    iree_interval_list_interval_data_t* this_data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(begin);
    if (!run) {
      run = true;
      fprintf(stderr, "------ %lu\n", this_data->offset);
    } else if (last_end < this_data->offset) {
      fprintf(stderr, "  xx   \n");
      fprintf(stderr, "  xx   \n");
      fprintf(stderr, "------ %lu\n", this_data->offset);
    } else if (last_end != this_data->offset) {
      fprintf(stderr, "!!!!!!! %lu", this_data->offset);
    }
    fprintf(stderr, "  ||  \n");
    if (print_element) {
      fprintf(stderr, "  ||  [%lu] ", this_data->size);
      print_element(stderr, this_data->data, user_data);
      fprintf(stderr, "\n");
    } else {
      fprintf(stderr, "  ||  [%lu]\n", this_data->size);
    }

    fprintf(stderr, "  ||  \n");
    fprintf(stderr, "------ %lu\n", this_data->offset + this_data->size);
    last_end = this_data->offset + this_data->size;
    begin = iree_tree_node_next(begin);
  }
}

void iree_interval_list_print(iree_interval_list_t* list,
                              void (*print_element)(FILE*, void*, void*),
                              void* user_data) {
  iree_interval_print(iree_tree_first(list), NULL, print_element, user_data,
                      false);
}

// Works like iree_interval_list_find but end is the last element,
// rather than one past the end.
void iree_interval_list_find_inclusive(iree_interval_list_t* interval_list,
                                       uint64_t offset, uint64_t size,
                                       iree_interval_t** begin,
                                       iree_interval_t** end) {
  if (!iree_tree_size(interval_list)) {
    *begin = *end = NULL;
    return;
  }
  *begin = iree_tree_lower_bound(interval_list, offset);
  *end = NULL;
  if (!*begin) {
    *begin = iree_tree_last(interval_list);
    iree_interval_list_interval_data_t* begin_data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(*begin);
    if (begin_data->offset + begin_data->size < offset) {
      *begin = NULL;
      return;
    }
  } else {
    iree_interval_t* prev = iree_tree_node_prev(*begin);
    if (prev) {
      iree_interval_list_interval_data_t* prev_data =
          (iree_interval_list_interval_data_t*)iree_tree_node_get_data(prev);
      if (prev_data->offset + prev_data->size > offset) {
        *begin = prev;
      }
    }
  }

  if (!*begin) {
    return;
  }

  iree_interval_list_interval_data_t* begin_data =
      (iree_interval_list_interval_data_t*)iree_tree_node_get_data(*begin);
  if (begin_data->offset >= offset + size) {
    *begin = NULL;
    return;
  }
  *end = iree_tree_upper_bound(interval_list, offset + size);
  if (*end == NULL) {
    *end = iree_tree_last(interval_list);
  } else {
    *end = iree_tree_node_prev(*end);
  }
}

iree_status_t iree_interval_list_insert(iree_interval_list_t* interval_list,
                                        uint64_t offset, uint64_t size,
                                        iree_interval_t** out) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot insert an empty interval");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_interval_list_erase(interval_list, offset, size));
  iree_interval_t* inserted;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tree_insert(interval_list, offset, &inserted));
  iree_interval_list_interval_data_t* new_data =
      (iree_interval_list_interval_data_t*)iree_tree_node_get_data(inserted);
  new_data->offset = offset;
  new_data->size = size;
  *out = inserted;
  return iree_ok_status();
}

iree_status_t iree_interval_list_insert_no_overwrite(
    iree_interval_list_t* interval_list, uint64_t offset, uint64_t size,
    iree_interval_t** out_begin, iree_interval_t** out_end) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot insert an empty interval");
  }

  iree_interval_t* first_affected_interval;
  iree_interval_t* last_affected_interval;
  iree_interval_list_find_inclusive(interval_list, offset, size,
                                    &first_affected_interval,
                                    &last_affected_interval);
  // We couldn't actually find anything in range.
  if (!first_affected_interval) {
    iree_interval_t* inserted;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tree_insert(interval_list, offset, &inserted));
    iree_interval_list_interval_data_t* new_data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(inserted);
    new_data->offset = offset;
    new_data->size = size;
    *out_begin = inserted;
    *out_end = iree_tree_node_next(inserted);
    return iree_ok_status();
  }

  iree_interval_t* interval_walk = first_affected_interval;
  iree_host_size_t loops = 0;
  do {
    iree_interval_list_interval_data_t* data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
            interval_walk);
    // Our outer region overlaps us entirely, so we end up with 3 regions.
    if (data->offset < offset && data->offset + data->size > offset + size) {
      IREE_ASSERT(interval_walk == last_affected_interval);
      IREE_ASSERT(loops == 0);
      // Shrink the region
      iree_host_size_t after_size = data->offset + data->size - (offset + size);
      data->size = offset - data->offset;
      {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list, offset, &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        new_data->offset = offset;
        new_data->size = size;
        memcpy(new_data->data, data->data,
               iree_tree_get_data_size(interval_list) -
                   sizeof(iree_interval_list_interval_data_t));
        first_affected_interval = inserted;
        last_affected_interval = inserted;
      }
      {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list, offset + size, &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        new_data->offset = offset + size;
        new_data->size = after_size;
        memcpy(new_data->data, data->data,
               iree_tree_get_data_size(interval_list) -
                   sizeof(iree_interval_list_interval_data_t));
      }
      break;
    } else if (data->offset >= offset &&
               data->offset + data->size <= offset + size) {
      if (interval_walk == last_affected_interval) {
        break;
      }
      interval_walk = iree_tree_node_next(interval_walk);
    } else if (data->offset < offset &&
               data->offset + data->size <= offset + size) {
      IREE_ASSERT(loops == 0);
      // This element overlaps, but is off the lhs of the one
      // we want to insert, split it.
      {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list, offset, &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        memcpy(new_data->data, data->data,
               iree_tree_get_data_size(interval_list) -
                   sizeof(iree_interval_list_interval_data_t));
        new_data->offset = offset;
        new_data->size = data->offset + data->size - offset;
        // Move the first affected interval up one, since
        // we split at the insertion line
        if (last_affected_interval == first_affected_interval) {
          last_affected_interval = inserted;
          interval_walk = last_affected_interval;
        }
        first_affected_interval = inserted;
      }
      data->size = offset - data->offset;
      if (interval_walk == last_affected_interval) {
        break;
      }
      interval_walk = iree_tree_node_next(interval_walk);
    } else if (data->offset >= offset &&
               data->offset + data->size > offset + size) {
      // This element overlaps, but is off the rhs of the one we
      // want to insert, split it.
      iree_host_size_t new_offset = data->offset;
      iree_host_size_t new_size = offset + size - data->offset;

      IREE_ASSERT(interval_walk == last_affected_interval);
      data->size = data->offset + data->size - (offset + size);
      data->offset = offset + size;
      iree_tree_move_node(interval_list, interval_walk, data->offset);
      {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list, new_offset, &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        memcpy(new_data->data, data->data,
               iree_tree_get_data_size(interval_list) -
                   sizeof(iree_interval_list_interval_data_t));
        new_data->offset = new_offset;
        new_data->size = new_size;
        // Move the last affected interval back one, since
        // we split at the insertion line
        if (last_affected_interval == first_affected_interval) {
          first_affected_interval = inserted;
        }
        last_affected_interval = inserted;
      }
      break;
    }
    loops++;
  } while (true);

  // First add anything before the first interval if needed.
  {
    iree_interval_list_interval_data_t* data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
            first_affected_interval);
    if (data->offset > offset) {
      iree_interval_t* inserted;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_tree_insert(interval_list, offset, &inserted));
      iree_interval_list_interval_data_t* new_data =
          (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
              inserted);
      new_data->offset = offset;
      new_data->size = data->offset - offset;
      first_affected_interval = inserted;
    }
  }
  // Next add anything after the last interval if needed
  {
    iree_interval_list_interval_data_t* data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
            last_affected_interval);
    if (data->offset + data->size < offset + size) {
      iree_interval_t* inserted;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_tree_insert(interval_list, data->offset + data->size,
                               &inserted));
      iree_interval_list_interval_data_t* new_data =
          (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
              inserted);
      new_data->offset = data->offset + data->size;
      new_data->size = offset + size - (data->offset + data->size);
      last_affected_interval = inserted;
    }
  }

  if (first_affected_interval != last_affected_interval) {
    interval_walk = first_affected_interval;
    // Now fill in the gaps
    iree_interval_list_interval_data_t* this_data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
            interval_walk);
    do {
      interval_walk = iree_tree_node_next(interval_walk);
      iree_interval_list_interval_data_t* previous_data = this_data;
      this_data = (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
          interval_walk);
      if (this_data->offset != previous_data->offset + previous_data->size) {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list,
                                 previous_data->offset + previous_data->size,
                                 &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        new_data->offset = previous_data->offset + previous_data->size;
        new_data->size = this_data->offset - new_data->offset;
      }
    } while (interval_walk != last_affected_interval);
  }

  *out_begin = first_affected_interval;
  *out_end = iree_tree_node_next(last_affected_interval);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_interval_list_find(iree_interval_list_t* interval_list,
                             uint64_t offset, uint64_t size,
                             iree_interval_t** begin, iree_interval_t** end) {
  if (!iree_tree_size(interval_list)) {
    *begin = *end = NULL;
    return;
  }
  *begin = iree_tree_lower_bound(interval_list, offset);
  *end = NULL;
  if (!*begin) {
    *begin = iree_tree_last(interval_list);
    iree_interval_list_interval_data_t* begin_data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(*begin);
    if (begin_data->offset + begin_data->size < offset) {
      *begin = NULL;
      return;
    }
  } else {
    iree_interval_t* prev = iree_tree_node_prev(*begin);
    if (prev) {
      iree_interval_list_interval_data_t* prev_data =
          (iree_interval_list_interval_data_t*)iree_tree_node_get_data(prev);
      if (prev_data->offset + prev_data->size > offset) {
        *begin = prev;
      }
    }
  }

  if (!*begin) {
    return;
  }

  iree_interval_list_interval_data_t* begin_data =
      (iree_interval_list_interval_data_t*)iree_tree_node_get_data(*begin);
  if (begin_data->offset > offset + size) {
    *begin = NULL;
    return;
  }
  *end = iree_tree_lower_bound(interval_list, offset + size);
}

iree_status_t iree_interval_list_erase(iree_interval_list_t* interval_list,
                                       uint64_t offset, uint64_t size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_interval_t* first_affected_interval;
  iree_interval_t* last_affected_interval;
  iree_interval_list_find_inclusive(interval_list, offset, size,
                                    &first_affected_interval,
                                    &last_affected_interval);
  // We couldn't actually find anything in range.
  if (!first_affected_interval) {
    return iree_ok_status();
  }

  iree_host_size_t loops = 0;
  do {
    iree_interval_list_interval_data_t* data =
        (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
            first_affected_interval);
    // The outer region overlaps us entirely, resize the left, and
    // create a new right around the hole.
    if (data->offset < offset && data->offset + data->size > offset + size) {
      IREE_ASSERT(first_affected_interval == last_affected_interval);
      IREE_ASSERT(loops == 0);
      // Shrink the region
      iree_host_size_t after_size = data->offset + data->size - (offset + size);
      data->size = offset - data->offset;
      {
        iree_interval_t* inserted;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_tree_insert(interval_list, offset + size, &inserted));
        iree_interval_list_interval_data_t* new_data =
            (iree_interval_list_interval_data_t*)iree_tree_node_get_data(
                inserted);
        new_data->offset = offset + size;
        new_data->size = after_size;
        memcpy(new_data->data, data->data,
               iree_tree_get_data_size(interval_list) -
                   sizeof(iree_interval_list_interval_data_t));
      }
      break;
    } else if (data->offset >= offset &&
               data->offset + data->size <= offset + size) {
      iree_interval_t* to_remove = first_affected_interval;
      first_affected_interval = iree_tree_node_next(first_affected_interval);
      // The new node is entirely contained within our node. Remove it.
      iree_tree_erase(interval_list, to_remove);
      if (to_remove == last_affected_interval) {
        break;
      }
    } else if (data->offset < offset &&
               data->offset + data->size <= offset + size) {
      IREE_ASSERT(loops == 0);
      // Just shrink the element.
      data->size = offset - data->offset;
      if (first_affected_interval == last_affected_interval) {
        break;
      }
      first_affected_interval = iree_tree_node_next(first_affected_interval);
    } else if (data->offset >= offset &&
               data->offset + data->size > offset + size) {
      IREE_ASSERT(first_affected_interval == last_affected_interval);
      data->size = data->size - (offset + size - data->offset);
      data->offset = offset + size;
      iree_tree_move_node(interval_list, first_affected_interval, data->offset);
      break;
    }
    loops++;
  } while (true);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
