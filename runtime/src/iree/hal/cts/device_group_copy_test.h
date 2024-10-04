// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_DEVICE_GROUP_COPY_TEST_H_
#define IREE_HAL_CTS_DEVICE_GROUP_COPY_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kDefaultAllocationSize = 1024;
}  // namespace

class DeviceGroupCopyTest : public CTSTestBase<> {};

TEST_F(DeviceGroupCopyTest, CopyBetweenDevicesFromQueue0) {
  iree_hal_command_buffer_t* command_buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x1,
      /*binding_capacity=*/0, &command_buffer1));

  iree_hal_command_buffer_t* command_buffer2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x1,
      /*binding_capacity=*/0, &command_buffer2));
  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), i8_val, kDefaultAllocationSize);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  host_params.queue_affinity = 0x1;
  iree_hal_buffer_t* host_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, kDefaultAllocationSize, &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, reference_buffer.data(), host_buffer, 0, reference_buffer.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  // Create a device buffer.
  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  device_params.queue_affinity = 0x1 | 0x2;

  iree_hal_buffer_t* device_buffer_device_1 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_1));

  device_params.queue_affinity = 0x2;
  iree_hal_buffer_t* device_buffer_device_2 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_2));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer1));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer1, /*source_ref=*/
      iree_hal_make_buffer_ref(host_buffer, 0, kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer1));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer1));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer2));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer2, /*source_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_2, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer2));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer2));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer_device_2, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_buffer_release(device_buffer_device_1);
  iree_hal_buffer_release(device_buffer_device_2);
  iree_hal_buffer_release(host_buffer);
}

TEST_F(DeviceGroupCopyTest, CopyBetweenDevicesFromQueue1) {
  iree_hal_command_buffer_t* command_buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x2,
      /*binding_capacity=*/0, &command_buffer1));

  iree_hal_command_buffer_t* command_buffer2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x2,
      /*binding_capacity=*/0, &command_buffer2));
  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), i8_val, kDefaultAllocationSize);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  host_params.queue_affinity = 0x1;
  iree_hal_buffer_t* host_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, kDefaultAllocationSize, &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, reference_buffer.data(), host_buffer, 0, reference_buffer.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  // Create a device buffer.
  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  device_params.queue_affinity = 0x1 | 0x2;

  iree_hal_buffer_t* device_buffer_device_1 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_1));

  device_params.queue_affinity = 0x2;
  iree_hal_buffer_t* device_buffer_device_2 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_2));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer1));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer1, /*source_ref=*/
      iree_hal_make_buffer_ref(host_buffer, 0, kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer1));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      command_buffer1, iree_hal_buffer_binding_table_empty(), 0x2));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer2));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer2, /*source_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_2, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer2));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      command_buffer2, iree_hal_buffer_binding_table_empty(), 0x2));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer_device_2, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_buffer_release(device_buffer_device_1);
  iree_hal_buffer_release(device_buffer_device_2);
  iree_hal_buffer_release(host_buffer);
}

TEST_F(DeviceGroupCopyTest, CopyBetweenDevicesFromBothQueues) {
  iree_hal_command_buffer_t* command_buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x1,
      /*binding_capacity=*/0, &command_buffer1));

  iree_hal_command_buffer_t* command_buffer2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0x2,
      /*binding_capacity=*/0, &command_buffer2));
  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), i8_val, kDefaultAllocationSize);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  host_params.queue_affinity = 0x1;
  iree_hal_buffer_t* host_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, kDefaultAllocationSize, &host_buffer));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, reference_buffer.data(), host_buffer, 0, reference_buffer.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  // Create a device buffer.
  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  device_params.queue_affinity = 0x1 | 0x2;

  iree_hal_buffer_t* device_buffer_device_1 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_1));

  device_params.queue_affinity = 0x2;
  iree_hal_buffer_t* device_buffer_device_2 = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      &device_buffer_device_2));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer1));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer1, /*source_ref=*/
      iree_hal_make_buffer_ref(host_buffer, 0, kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer1));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      command_buffer1, iree_hal_buffer_binding_table_empty(), 0x1));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer2));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer2, /*source_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_1, 0,
                               kDefaultAllocationSize),
      /*target_ref=*/
      iree_hal_make_buffer_ref(device_buffer_device_2, 0,
                               kDefaultAllocationSize)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer2));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      command_buffer2, iree_hal_buffer_binding_table_empty(), 0x2));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer_device_2, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_buffer_release(device_buffer_device_1);
  iree_hal_buffer_release(device_buffer_device_2);
  iree_hal_buffer_release(host_buffer);
}

TEST_F(DeviceGroupCopyTest, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_DEVICE_GROUP_COPY_TEST_H_
