# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
# Currently only gfx9 and gfx12 archs are supported
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx12")
if(NOT INST_TARGETS)
  message(WARNING "Skipping SageAttention compilation: No supported GPU targets (gfx9, gfx12) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
  return()
endif()

# ====================================================================
# SageAttention codegen - only FWD API, minimal instances
# ====================================================================
file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

list(JOIN INST_TARGETS , SAGEATTENTION_TARGETS_ARG)

# Only generate FWD API, only common head dimensions (64, 128)
set(SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --targets ${SAGEATTENTION_TARGETS_ARG}
  --api sageattention_fwd
  --optdim 64,128
)

# Generate list of kernels to build
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/sageattention_fwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "SageAttention FAILED to generate kernel list via Python.")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/sageattention_fwd_blob_list.txt SAGEATTENTION_FWD_GEN_BLOBS)

# Generate the kernel instance files
add_custom_command(
  OUTPUT ${SAGEATTENTION_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate SageAttention FWD kernels"
  VERBATIM
)

# Build the kernel instances library
add_library(tile_sageattention_fwd_instances OBJECT ${SAGEATTENTION_FWD_GEN_BLOBS})
target_include_directories(tile_sageattention_fwd_instances PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Compile options for kernel instances
set(SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -Wno-undefined-func-template)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_API=1)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=0)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=0)
list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_PAGEDKV_API=0)

if(NOT DEFINED SAGEATTENTION_FWD_FAST_EXP2)
  set(SAGEATTENTION_FWD_FAST_EXP2 ON)
endif()

if(SAGEATTENTION_FWD_FAST_EXP2)
  list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
  list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()

if(CK_USE_OCP_FP8)
  list(APPEND SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(tile_sageattention_fwd_instances PRIVATE ${SAGEATTENTION_FWD_INSTANCE_COMPILE_OPTIONS})
set_property(TARGET tile_sageattention_fwd_instances PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})
set_property(TARGET tile_sageattention_fwd_instances PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# SageAttention FWD Example
# ====================================================================
set(EXAMPLE_SAGEATTENTION_FWD "tile_example_sageattention_fwd")

message(DEBUG "adding example ${EXAMPLE_SAGEATTENTION_FWD}")

add_executable(${EXAMPLE_SAGEATTENTION_FWD} EXCLUDE_FROM_ALL example_sageattention_fwd.cpp)
target_include_directories(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Link with our own minimal instances library (INDEPENDENT from FMHA!)
target_link_libraries(${EXAMPLE_SAGEATTENTION_FWD} tile_sageattention_fwd_instances)

set(SAGEATTENTION_FWD_COMPILE_OPTIONS)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -Wno-undefined-func-template)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_API=1)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=0)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=0)
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_PAGEDKV_API=0)

if(SAGEATTENTION_FWD_FAST_EXP2)
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()

if(CK_USE_OCP_FP8)
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE ${SAGEATTENTION_FWD_COMPILE_OPTIONS})
set_property(TARGET ${EXAMPLE_SAGEATTENTION_FWD} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})
