// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_debug_info.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/hip_executable_def_reader.h"
#include "iree/schemas/hip_executable_def_verifier.h"

typedef struct iree_hal_hip_native_executable_per_device_data_t {

  // Loaded HIP modules.
  iree_host_size_t module_count;
  hipModule_t* modules;
  
  // Exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  iree_hal_hip_kernel_params_t exports[];
} iree_hal_hip_native_executable_per_device_data_t;

typedef struct iree_hal_hip_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  const iree_hal_hip_dynamic_symbols_t* symbols;

  uint32_t num_devices;
  iree_host_size_t native_info_device_size;
  iree_hal_hip_native_executable_per_device_data_t* per_device_data;

} iree_hal_hip_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hip_native_executable_vtable;

static iree_hal_hip_native_executable_t* iree_hal_hip_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_native_executable_vtable);
  return (iree_hal_hip_native_executable_t*)base_value;
}

typedef struct iree_hal_hip_limits_t {
  uint32_t max_block_dims[3];
  uint32_t max_block_shared_memory_size;
} iree_hal_hip_limits_t;
static iree_status_t iree_hal_hip_query_limits(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_hal_hip_limits_t* out_limits) {
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[0],
                            hipDeviceAttributeMaxBlockDimX, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[1],
                            hipDeviceAttributeMaxBlockDimY, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_dims[2],
                            hipDeviceAttributeMaxBlockDimZ, device),
      "hipDeviceGetAttribute");

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute(&out_limits->max_block_shared_memory_size,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, device),
      "hipDeviceGetAttribute");

  return iree_ok_status();
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_hip_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    const iree_hal_hip_limits_t* limits) {
  if (!flatbuffer_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present");
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_hal_hip_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_hip_ModuleDef_table_t module_def =
        iree_hal_hip_ModuleDef_vec_at(modules_vec, i);
    if (!module_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] is NULL", i);
    }
    if (flatbuffers_string_len(
            iree_hal_hip_ModuleDef_hsaco_image_get(module_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] contents are empty", i);
    }
  }

  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  for (iree_host_size_t i = 0; i < iree_hal_hip_ExportDef_vec_len(exports_vec);
       ++i) {
    iree_hal_hip_ExportDef_table_t export_def =
        iree_hal_hip_ExportDef_vec_at(exports_vec, i);
    if (!export_def) continue;

    uint32_t module_ordinal =
        iree_hal_hip_ExportDef_module_ordinal_get(export_def);
    if (module_ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz
                              "] module_ordinal %u is out of bounds %" PRIhsz,
                              i, module_ordinal, module_count);
    }

    if (flatbuffers_string_len(
            iree_hal_hip_ExportDef_kernel_name_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] name is empty", i);
    }

    if (iree_hal_hip_ExportDef_block_dims_is_present(export_def)) {
      const iree_hal_hip_BlockDims_t* block_dims =
          iree_hal_hip_ExportDef_block_dims_get(export_def);
      if (block_dims->x > limits->max_block_dims[0] ||
          block_dims->y > limits->max_block_dims[1] ||
          block_dims->z > limits->max_block_dims[2]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz
            "] block dims %ux%ux%u exceeds device maximum %ux%ux%u",
            i, block_dims->x, block_dims->y, block_dims->z,
            limits->max_block_dims[0], limits->max_block_dims[1],
            limits->max_block_dims[2]);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] blocks dims are missing",
                              i);
    }

    uint32_t block_shared_memory_size =
        iree_hal_hip_ExportDef_block_shared_memory_size_get(export_def);
    if (block_shared_memory_size > limits->max_block_shared_memory_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz
                              "] requires %uB of shared memory and "
                              "exceeds the device maximum of %uB per block",
                              i, block_shared_memory_size,
                              limits->max_block_shared_memory_size);
    }

    uint32_t constant_count =
        iree_hal_hip_ExportDef_constant_count_get(export_def);
    if (constant_count > IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
          constant_count, IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT);
    }

    iree_hal_hip_BindingBits_vec_t binding_flags_vec =
        iree_hal_hip_ExportDef_binding_flags_get(export_def);
    if (iree_hal_hip_BindingBits_vec_len(binding_flags_vec) >
        IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] binding_flags count %zu exceeds maximum of %u",
          i, iree_hal_hip_BindingBits_vec_len(binding_flags_vec),
          IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_hip_ExportDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hip_native_executable_create(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    uint32_t num_devices, hipDevice_t* devices,
    hipCtx_t* contexts,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (num_devices < 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "Need at least one device");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;

  // TODO: move to the executable cache to avoid repeated queries.
  iree_hal_hip_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_query_limits(symbols, devices[0], &limits));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_flatbuffer_verify(
              executable_params->executable_data, &limits));

  iree_hal_hip_ExecutableDef_table_t executable_def =
      iree_hal_hip_ExecutableDef_as_root(
          executable_params->executable_data.data);

  iree_hal_hip_ModuleDef_vec_t modules_vec =
      iree_hal_hip_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(modules_vec);
  iree_hal_hip_ExportDef_vec_t exports_vec =
      iree_hal_hip_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count = iree_hal_hip_ExportDef_vec_len(exports_vec);

  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_export_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_hip_ExportDef_table_t export_def =
          iree_hal_hip_ExportDef_vec_at(exports_vec, i);
      total_export_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_hip_ExportDef_debug_info_get(export_def));
    }
  });

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hip_native_executable_t* executable = NULL;
  const iree_host_size_t native_executable_device_info_size = 
     module_count * sizeof(executable->per_device_data[0]) + export_count * sizeof(executable->per_device_data->exports[0]) + total_export_info_length;
  const iree_host_size_t total_size =
      sizeof(*executable) + num_devices * native_executable_device_info_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_hip_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->symbols = symbols;
  executable->native_info_device_size = native_executable_device_info_size;
  executable->num_devices = num_devices;
  iree_status_t status = iree_ok_status();


  for (size_t j = 0; j < num_devices && iree_status_is_ok(status); ++j) {
    status = IREE_HIP_RESULT_TO_STATUS(symbols,
            hipCtxPushCurrent(contexts[j]),
            "hipCtxPushCurrent");
    
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        break;
    }
    iree_hal_hip_native_executable_per_device_data_t* ned = 
        (iree_hal_hip_native_executable_per_device_data_t*)((uint8_t*)(executable->per_device_data) + j * native_executable_device_info_size);
        
    ned->module_count = module_count;
    ned->modules =
        (hipModule_t*)((uint8_t*)ned + sizeof(*ned) +
                        (export_count * sizeof(ned->exports[0])));
    ned->export_count = export_count;
    IREE_TRACE(
        iree_hal_debug_export_info_t* export_infos =
            (iree_hal_debug_export_info_t*)((uint8_t*)ned->modules +
                                            module_count * 
                                                sizeof(ned->modules[0]) * num_devices));

    // Publish any embedded source files to the tracing infrastructure.
    iree_hal_debug_publish_source_files(
        iree_hal_hip_ExecutableDef_source_files_get(executable_def));

    // Load each module first so that exports can reference them.
    for (iree_host_size_t i = 0; i < module_count; ++i) {
        iree_hal_hip_ModuleDef_table_t module_def =
            iree_hal_hip_ModuleDef_vec_at(modules_vec, i);

        // WARNING: HIP doesn't take an expected length here so we can't bound it.
        // It's likely that users could craft inputs that read beyond the extents of
        // the embedded binary.
        flatbuffers_string_t hsaco_image =
            iree_hal_hip_ModuleDef_hsaco_image_get(module_def);

        // TODO: pass hipJitOption values to get log info and other info back.
        // We pass the error buffer today but could use the info log to diagnose
        // performance warnings.
        char error_log[8192] = {0};
        hipJitOption jit_options[] = {
            hipJitOptionErrorLogBuffer,
            hipJitOptionErrorLogBufferSizeBytes,
        };
        void* jit_option_values[] = {
            (void*)error_log,
            (void*)(uint32_t)sizeof(error_log),
        };
        hipModule_t module = NULL;
        status = IREE_HIP_RESULT_TO_STATUS(
            symbols,
            hipModuleLoadDataEx(&module, hsaco_image, IREE_ARRAYSIZE(jit_options),
                                jit_options, jit_option_values),
            "hipModuleLoadDataEx");
        if (!iree_status_is_ok(status)) {
            status = iree_status_annotate(
                status,
                IREE_SV("mismatched target chip? missing/wrong bitcode directory?"));
            if (strlen(error_log) > 0) {
                status =
                    iree_status_annotate(status, iree_make_cstring_view(error_log));
            }
            break;
        }

        ned->modules[i] = module;
    

        if (iree_status_is_ok(status)) {
            for (iree_host_size_t i = 0; i < export_count; ++i) {
                iree_hal_hip_ExportDef_table_t export_def =
                    iree_hal_hip_ExportDef_vec_at(exports_vec, i);

                // Lookup the function in the module; this should always succeed but
                // we cannot trust that the input was generated by our compiler.
                uint32_t module_ordinal =
                    iree_hal_hip_ExportDef_module_ordinal_get(export_def);
                hipModule_t module = ned->modules[module_ordinal];
                flatbuffers_string_t kernel_name =
                    iree_hal_hip_ExportDef_kernel_name_get(export_def);
                hipFunction_t function = NULL;
                status = IREE_HIP_RESULT_TO_STATUS(
                    symbols, hipModuleGetFunction(&function, module, kernel_name),
                    "hipModuleGetFunction");
                if (!iree_status_is_ok(status)) break;
                if (!function) {
                    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                            "exports[%" PRIhsz
                                            "] kernel `%s` not found in modules[%u]",
                                            i, kernel_name, module_ordinal);
                    break;
                }

                uint32_t block_shared_memory_size =
                    iree_hal_hip_ExportDef_block_shared_memory_size_get(export_def);
                status = IREE_HIP_RESULT_TO_STATUS(
                    symbols,
                    hipFuncSetAttribute(
                        function,
                        (hipFuncAttribute)
                            HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        block_shared_memory_size),
                    "hipFuncSetAttribute");
                if (!iree_status_is_ok(status)) break;

                // Package required parameters for kernel launches for each entry point.
                iree_hal_hip_kernel_params_t* kernel_info = &ned->exports[i];
                kernel_info->function = function;
                const iree_hal_hip_BlockDims_t* block_dims =
                    iree_hal_hip_ExportDef_block_dims_get(export_def);
                kernel_info->block_dims[0] = block_dims->x;
                kernel_info->block_dims[1] = block_dims->y;
                kernel_info->block_dims[2] = block_dims->z;
                kernel_info->block_shared_memory_size =
                    iree_hal_hip_ExportDef_block_shared_memory_size_get(export_def);
                kernel_info->constant_count =
                    iree_hal_hip_ExportDef_constant_count_get(export_def);
                iree_hal_hip_BindingBits_vec_t binding_flags_vec =
                    iree_hal_hip_ExportDef_binding_flags_get(export_def);
                kernel_info->binding_count =
                    iree_hal_hip_BindingBits_vec_len(binding_flags_vec);

                IREE_TRACE({
                    iree_hal_debug_copy_export_info(
                        iree_hal_hip_ExportDef_debug_info_get(export_def),
                        &export_infos[i]);
                    kernel_info->debug_info.function_name = export_infos[i].function_name;
                    kernel_info->debug_info.source_filename =
                        export_infos[i].source_filename;
                    kernel_info->debug_info.source_line = export_infos[i].source_line;
                });
            }
        }
    }
    {
        IREE_IGNORE_ERROR(
            IREE_HIP_RESULT_TO_STATUS(symbols,
                hipCtxPopCurrent(NULL), "hipCtxPopCurrent"));
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hip_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t j = 0; j < executable->num_devices; ++j) {
    for (iree_host_size_t i = 0; i < executable->per_device_data[j].module_count; ++i) {
        if (executable->per_device_data[j].modules[i]) {
        IREE_HIP_IGNORE_ERROR(executable->symbols,
                                hipModuleUnload(executable->per_device_data[j].modules[i]));
        }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable, int32_t ordinal,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_kernel_params_t** out_params) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  int device_idx = 0;
  if (queue_affinity) {
     device_idx = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_idx > executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "Invalid affinity");
  }
    
  if (ordinal >= executable->per_device_data[device_idx].export_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "export ordinal %d out of range; executable contains %" PRIhsz
        " exports",
        ordinal, executable->per_device_data[device_idx].export_count);
  }
  *out_params = &executable->per_device_data[device_idx].exports[ordinal];
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_hip_native_executable_vtable = {
        .destroy = iree_hal_hip_native_executable_destroy,
};
