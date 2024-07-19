#include "iree/base/circular_array.h"

typedef struct iree_circular_array_t {
  size_t capacity;
  size_t count;
  size_t element_size;
  uint8_t *head;
  uint8_t *tail;
  iree_allocator_t host_allocator;
  uint8_t elements[];
} iree_circular_array_t;

iree_status_t iree_circular_array_create(iree_allocator_t host_allocator,
                                         iree_host_size_t capacity,
                                         iree_host_size_t element_size,
                                         iree_circular_array_t **out) {
  iree_circular_array_t *t;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(iree_circular_array_t) + capacity * ((element_size + 7) & ~7),
      (void **)&t);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  t->capacity = capacity;
  t->count = 0;
  t->element_size = element_size;
  t->head = t->elements;
  t->tail = t->elements;
  t->host_allocator = host_allocator;
  *out = t;
  return iree_ok_status();
}

void iree_circular_array_free(iree_circular_array_t *array) {
  iree_allocator_free(array->host_allocator, array);
}

iree_status_t iree_circular_array_push_back(iree_circular_array_t *array,
                                            void **out_item) {
  if (array->count == array->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "circular array out of space");
    // handle error
  }
  *out_item = array->head;
  array->head = array->head + (((array->element_size + 7) & ~7));
  if (array->head ==
      array->elements + (array->capacity * ((array->element_size + 7) & ~7))) {
    array->head = array->elements;
  }
  array->count += 1;
  return iree_ok_status();
}

void iree_circular_array_front(iree_circular_array_t *array, void **out_item) {
  *out_item = array->tail;
}

void iree_circular_array_pop_front(iree_circular_array_t *array) {
  array->tail = array->tail + (((array->element_size + 7) & ~7));
  if (array->tail ==
      array->elements + (array->capacity * ((array->element_size + 7) & ~7))) {
    array->tail = array->elements;
  }
  array->count--;
}

iree_host_size_t iree_circular_array_size(iree_circular_array_t *array) {
  return array->count;
}
