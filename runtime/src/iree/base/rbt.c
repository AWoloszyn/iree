// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/rbt.h"

#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/config.h"
#include "stdint.h"

typedef struct iree_tree_node_t iree_tree_node_t;
typedef struct iree_tree_node_t {
  bool red;
  iree_tree_node_t* left;
  iree_tree_node_t* right;
  iree_tree_node_t* parent;
  iree_host_size_t key;
  bool is_sentinel;
  uint8_t data[];
} iree_tree_node_t;

typedef struct iree_tree_t {
  uint32_t element_size;
  iree_allocator_t allocator;
  iree_tree_node_t* root;
  iree_host_size_t size;
  iree_tree_node_t nil;
} iree_tree_t;

iree_status_t iree_tree_create(iree_allocator_t allocator,
                               iree_host_size_t element_size,
                               iree_tree_t** out) {
  iree_tree_t* t;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(iree_tree_t), (void**)&t);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    return status;
  }
  t->element_size = element_size;
  t->allocator = allocator;
  t->root = &t->nil;
  t->size = 0;
  t->nil.is_sentinel = true;
  *out = t;
  return status;
}

iree_host_size_t iree_tree_size(iree_tree_t* tree) { return tree->size; }

bool iree_tree_free_node(iree_tree_node_t* node, void* user_data) {
  iree_tree_t* tree = (iree_tree_t*)user_data;
  iree_allocator_free(tree->allocator, node);
  return true;
}

void iree_tree_free(iree_tree_t* tree) {
  iree_tree_walk(tree, iree_tree_free_node, tree);
}

void iree_tree_rotate_left(iree_tree_t* tree, iree_tree_node_t* x) {
  iree_tree_node_t* y = x->right;
  x->right = y->left;
  if (y->left != &tree->nil) {
    y->left->parent = x;
  }
  y->parent = x->parent;
  if (x->parent == NULL) {
    tree->root = y;
  } else if (x == x->parent->left) {
    x->parent->left = y;
  } else {
    x->parent->right = y;
  }
  y->left = x;
  x->parent = y;
}

void iree_tree_rotate_right(iree_tree_t* tree, iree_tree_node_t* x) {
  iree_tree_node_t* y = x->left;
  x->left = y->right;
  if (y->right != &tree->nil) {
    y->right->parent = x;
  }
  y->parent = x->parent;
  if (x->parent == NULL) {
    tree->root = y;
  } else if (x == x->parent->right) {
    x->parent->right = y;
  } else {
    x->parent->left = y;
  }
  y->right = x;
  x->parent = y;
}

iree_status_t iree_tree_insert_internal(iree_tree_t* tree, iree_host_size_t key,
                                        iree_tree_node_t* t) {
  t->left = &tree->nil;
  t->right = &tree->nil;
  t->key = key;
  t->red = true;  // red
  t->parent = NULL;

  iree_tree_node_t* x = tree->root;
  iree_tree_node_t* y = NULL;
  while (x != &tree->nil) {
    y = x;
    if (t->key < x->key) {
      x = x->left;
    } else if (t->key > x->key) {
      x = x->right;
    } else {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "Trying to insert a duplicate key");
    }
  }
  t->parent = y;

  if (!y) {
    tree->root = t;
  } else if (t->key < y->key) {
    y->left = t;
  } else {
    y->right = t;
  }

  if (t->parent == NULL) {
    t->red = false;
    return iree_ok_status();
  }

  if (t->parent == tree->root) {
    return iree_ok_status();
  }

  while (t->parent->red) {
    if (t->parent == t->parent->parent->right) {
      iree_tree_node_t* uncle = t->parent->parent->left;
      if (uncle->red) {
        uncle->red = false;
        t->parent->red = false;
        t->parent->parent->red = true;
        t = t->parent->parent;
      } else {
        if (t == t->parent->left) {
          t = t->parent;
          iree_tree_rotate_right(tree, t);
        }
        t->parent->red = false;
        t->parent->parent->red = true;
        iree_tree_rotate_left(tree, t->parent->parent);
      }
    } else {
      iree_tree_node_t* uncle = t->parent->parent->right;
      if (uncle && uncle->red) {
        uncle->red = false;
        t->parent->red = false;
        t->parent->parent->red = true;
        t = t->parent->parent;
      } else {
        if (t == t->parent->right) {
          t = t->parent;
          iree_tree_rotate_left(tree, t);
        }
        t->parent->red = false;
        t->parent->parent->red = true;
        iree_tree_rotate_right(tree, t->parent->parent);
      }
    }
    if (t == tree->root) {
      break;
    }
  }
  tree->root->red = false;

  return iree_ok_status();
}

iree_status_t iree_tree_insert(iree_tree_t* tree, iree_host_size_t key,
                               iree_tree_node_t** out_data) {
  iree_tree_node_t* t;
  iree_status_t status = iree_allocator_malloc(
      tree->allocator, sizeof(iree_tree_node_t) + tree->element_size,
      (void**)&t);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    return status;
  }
  status = iree_tree_insert_internal(tree, key, t);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    return status;
  }
  tree->size++;
  *out_data = t;
  return status;
}

bool iree_tree_search(iree_tree_t* tree, iree_host_size_t key,
                      void** out_data) {
  iree_tree_node_t* node = tree->root;
  while (node->is_sentinel == false) {
    if (key == node->key) {
      *out_data = node->data;
      return true;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  return false;
}

iree_tree_node_t* iree_tree_get(iree_tree_t* tree, iree_host_size_t key) {
  iree_tree_node_t* node = tree->root;
  while (node->is_sentinel == false) {
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  return NULL;
}

bool iree_tree_walk_helper(iree_tree_node_t* node,
                           bool (*callback)(iree_tree_node_t*, void*),
                           void* user_data) {
  if (!node || node->is_sentinel) {
    return true;
  }
  if (!iree_tree_walk_helper(node->left, callback, user_data)) {
    return false;
  }
  if (!callback(node, user_data)) {
    return false;
  }
  return iree_tree_walk_helper(node->right, callback, user_data);
}

void iree_tree_walk(iree_tree_t* tree,
                    bool (*callback)(iree_tree_node_t*, void*),
                    void* user_data) {
  iree_tree_walk_helper(tree->root, callback, user_data);
}

iree_tree_node_t* iree_tree_node_next(iree_tree_node_t* node) {
  IREE_ASSERT(node != NULL);
  // 1. to find the smallest thing on our right hand side.
  if (!node->right->is_sentinel) {
    node = node->right;
    while (!node->left->is_sentinel) {
      node = node->left;
    }
    return node;
  }

  // 2. Find the parent who is not on the right
  iree_tree_node_t* parent = node->parent;
  while (parent && node == parent->right) {
    node = parent;
    parent = node->parent;
  }
  return parent;
}

iree_tree_node_t* iree_tree_node_prev(iree_tree_node_t* node) {
  IREE_ASSERT(node != NULL);
  // 1. to find the largest thing on our left hand side.
  if (!node->left->is_sentinel) {
    node = node->left;
    while (!node->right->is_sentinel) {
      node = node->right;
    }
    return node;
  }

  // 2. Find the parent who is not on the left
  iree_tree_node_t* parent = node->parent;
  while (parent && node == parent->left) {
    node = parent;
    parent = node->parent;
  }
  return parent;
}

void* iree_tree_node_get_data(iree_tree_node_t* node) {
  IREE_ASSERT(node);
  return node->data;
}

iree_host_size_t iree_tree_node_get_key(iree_tree_node_t* node) {
  IREE_ASSERT(node);
  return node->key;
}

iree_tree_node_t* iree_tree_lower_bound(iree_tree_t* tree,
                                        iree_host_size_t key) {
  iree_tree_node_t* node = tree->root;
  iree_tree_node_t* last = NULL;
  while (node->is_sentinel == false) {
    last = node;
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if (!last || last->key > key) {
    return last;
  }
  return iree_tree_node_next(last);
}

iree_tree_node_t* iree_tree_upper_bound(iree_tree_t* tree,
                                        iree_host_size_t key) {
  iree_tree_node_t* node = tree->root;
  iree_tree_node_t* last = NULL;
  while (node->is_sentinel == false) {
    last = node;
    if (key == node->key) {
      return node;
    } else if (key < node->key) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if (!last || last->key > key) {
    return last;
  }
  while (last && last->key <= key) {
    last = iree_tree_node_next(last);
  }
  return last;
}

void iree_tree_replace(iree_tree_t* tree, iree_tree_node_t* dst,
                       iree_tree_node_t* src) {
  if (!dst->parent) {
    tree->root = src;
  } else if (dst == dst->parent->left) {
    dst->parent->left = src;
  } else {
    dst->parent->right = src;
  }
  src->parent = dst->parent;
}

void iree_tree_remove(iree_tree_t* tree, iree_tree_node_t* z) {
  iree_tree_node_t *x, *y = NULL;
  y = z;

  bool initial_red = y->red;
  if (z->left == &tree->nil) {
    x = z->right;
    iree_tree_replace(tree, z, z->right);
  } else if (z->right == &tree->nil) {
    x = z->left;
    iree_tree_replace(tree, z, z->left);
  } else {
    y = iree_tree_node_next(z);
    initial_red = y->red;
    x = y->right;
    if (y->parent == z) {
      x->parent = y;
    } else {
      iree_tree_replace(tree, y, y->right);
      y->right = z->right;
      y->right->parent = y;
    }

    iree_tree_replace(tree, z, y);
    y->left = z->left;
    y->left->parent = y;
    y->red = z->red;
  }
  if (!initial_red) {
    while (x != tree->root && !x->red) {
      if (x == x->parent->left) {
        iree_tree_node_t* s = x->parent->right;
        if (s->red) {
          s->red = false;
          x->parent->red = true;
          iree_tree_rotate_left(tree, x->parent);
          s = x->parent->right;
        }

        if (!s->left->red && !s->right->red) {
          s->red = true;
          x = x->parent;
        } else {
          if (!s->right->red) {
            s->left->red = false;
            s->red = true;
            iree_tree_rotate_right(tree, s);
            s = x->parent->right;
          }
          s->red = x->parent->red;
          x->parent->red = false;
          s->right->red = false;
          iree_tree_rotate_left(tree, x->parent);
          x = tree->root;
        }
      } else {
        iree_tree_node_t* s = x->parent->left;
        if (s->red) {
          s->red = false;
          x->parent->red = true;
          iree_tree_rotate_right(tree, x->parent);
          s = x->parent->left;
        }

        if (!s->left->red && !s->right->red) {
          s->red = true;
          x = x->parent;
        } else {
          if (!s->left->red) {
            s->right->red = false;
            s->red = true;
            iree_tree_rotate_left(tree, s);
            s = x->parent->left;
          }

          s->red = x->parent->red;
          x->parent->red = false;
          s->left->red = false;
          iree_tree_rotate_right(tree, x->parent);
          x = tree->root;
        }
      }
    }
    x->red = false;
  }
}

void iree_tree_erase(iree_tree_t* tree, iree_tree_node_t* node) {
  iree_tree_remove(tree, node);
  iree_allocator_free(tree->allocator, node);
  tree->size--;
}

iree_tree_node_t* iree_tree_first(iree_tree_t* tree) {
  if (!tree->root || tree->root->is_sentinel) {
    return NULL;
  }
  iree_tree_node_t* val = tree->root;
  while (!val->left->is_sentinel) {
    val = val->left;
  }
  return val;
}

iree_tree_node_t* iree_tree_last(iree_tree_t* tree) {
  if (!tree->root || tree->root->is_sentinel) {
    return NULL;
  }
  iree_tree_node_t* val = tree->root;
  while (!val->right->is_sentinel) {
    val = val->right;
  }
  return val;
}

uint32_t iree_tree_get_data_size(iree_tree_t* tree) {
  return tree->element_size;
}

iree_status_t iree_tree_move_node(iree_tree_t* tree, iree_tree_node_t* node,
                                  iree_host_size_t new_key) {
  iree_tree_node_t* next = iree_tree_node_next(node);
  iree_tree_node_t* prev = iree_tree_node_prev(node);
  if ((!next || next->key > new_key) && (!prev || prev->key < new_key)) {
    // This node isn't going to move, just update it's value.
    node->key = new_key;
    return iree_ok_status();
  }
  iree_tree_remove(tree, node);
  return iree_tree_insert_internal(tree, new_key, node);
}
