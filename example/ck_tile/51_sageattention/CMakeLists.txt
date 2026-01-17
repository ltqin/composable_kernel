# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
# Currently only gfx9 and gfx12 archs are supported
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx12")
if(NOT INST_TARGETS)
  message(WARNING "Skipping SageAttention compilation: No supported GPU targets (gfx9, gfx12) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
  return()
endif()

set(EXAMPLE_SAGEATTENTION_FWD "tile_example_sageattention_fwd")

message(DEBUG "adding example ${EXAMPLE_SAGEATTENTION_FWD}")

# SageAttention currently reuses FMHA kernel infrastructure
# This means we link to tile_fmha_fwd_instances which contains ~1400 kernel variants
# 
# To reduce compilation time, you can:
# 1. Modify example/ck_tile/01_fmha/CMakeLists.txt:
#    - Set FMHA_FWD_ENABLE_APIS to "fwd" only (line 14)
#    - Reduce --optdim to "64,128" only (line 50)
#    This will reduce instances from 1400 to ~100
#
# 2. Or disable BUILD_TESTING in your cmake configuration
#    This also reduces the number of generated instances

add_executable(${EXAMPLE_SAGEATTENTION_FWD} EXCLUDE_FROM_ALL example_sageattention_fwd.cpp)
target_include_directories(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Link with FMHA instances (contains all kernel variants)
target_link_libraries(${EXAMPLE_SAGEATTENTION_FWD} tile_fmha_fwd_instances)

# Define API macros (we only use basic FWD API)
target_compile_definitions(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE
  CK_TILE_FMHA_FWD_API=1
  CK_TILE_FMHA_FWD_APPENDKV_API=0
  CK_TILE_FMHA_FWD_SPLITKV_API=0
  CK_TILE_FMHA_FWD_PAGEDKV_API=0
)

set(SAGEATTENTION_FWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -Wno-undefined-func-template)

# Allow comparing floating points directly in order to check sentinel values
list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -Wno-float-equal)

# Fast exp2 mode
if(NOT DEFINED SAGEATTENTION_FWD_FAST_EXP2)
  set(SAGEATTENTION_FWD_FAST_EXP2 ON)
endif()

if(SAGEATTENTION_FWD_FAST_EXP2)
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()

# conditionally specify the use of OCP_FP8
if(CK_USE_OCP_FP8)
  list(APPEND SAGEATTENTION_FWD_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE ${SAGEATTENTION_FWD_COMPILE_OPTIONS})
set_property(TARGET ${EXAMPLE_SAGEATTENTION_FWD} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

