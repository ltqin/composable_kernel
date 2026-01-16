# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
# Currently only gfx9 and gfx12 archs are supported
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx12")
if(NOT INST_TARGETS)
  message(WARNING "Skipping SageAttention compilation: No supported GPU targets (gfx9, gfx12) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
  return()
endif()

# Validate user-specified sageattention_fwd API list
set(SAGEATTENTION_FWD_KNOWN_APIS "fwd")
set(SAGEATTENTION_FWD_ENABLE_APIS "fwd" CACHE STRING
  "semicolon-separated list of APIs to generate (${SAGEATTENTION_FWD_KNOWN_APIS}) & link, or \"all\".")

if(SAGEATTENTION_FWD_ENABLE_APIS STREQUAL "all")
  set(SAGEATTENTION_FWD_ENABLE_APIS ${SAGEATTENTION_FWD_KNOWN_APIS})
endif()

foreach(api ${SAGEATTENTION_FWD_ENABLE_APIS})
  if(NOT "${api}" IN_LIST SAGEATTENTION_FWD_KNOWN_APIS)
    message(FATAL_ERROR "${api} isn't a known api: ${SAGEATTENTION_FWD_KNOWN_APIS}.")
  endif()
endforeach()

# Collect all codegen scripts
file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
# Re-run execute_process `generate.py --list_blobs` if any of the codegen scripts change
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

list(JOIN INST_TARGETS , SAGEATTENTION_TARGETS_ARG)

string(REPLACE ";" "," SAGEATTENTION_FWD_APIS "${SAGEATTENTION_FWD_ENABLE_APIS}")
set(SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --targets ${SAGEATTENTION_TARGETS_ARG}
  --api ${SAGEATTENTION_FWD_APIS}
  --optdim 64,128,256
)

# Generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "CK Tile SageAttention FAILED to generate a list of FWD kernels via Python.")
endif()

# NOTE: for cmake, the SAGEATTENTION_FWD_GEN_BLOBS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt SAGEATTENTION_FWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${SAGEATTENTION_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SAGEATTENTION_FWD_CODE_GEN_COMMON_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile SageAttention FWD kernels"
)

# Create library for SageAttention forward instances
add_library(tile_sageattention_fwd_instances OBJECT ${SAGEATTENTION_FWD_GEN_BLOBS})
target_include_directories(tile_sageattention_fwd_instances PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(tile_sageattention_fwd_instances PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)

if(CK_USE_OCP_FP8)
  target_compile_definitions(tile_sageattention_fwd_instances PRIVATE CK_TILE_USE_OCP_FP8)
endif()

set_property(TARGET tile_sageattention_fwd_instances PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})
set_property(TARGET tile_sageattention_fwd_instances PROPERTY POSITION_INDEPENDENT_CODE ON)

# Create example executable
set(EXAMPLE_SAGEATTENTION_FWD "tile_example_sageattention_fwd")

message(DEBUG "adding example ${EXAMPLE_SAGEATTENTION_FWD}")

add_executable(${EXAMPLE_SAGEATTENTION_FWD} EXCLUDE_FROM_ALL example_sageattention_fwd.cpp)
target_include_directories(${EXAMPLE_SAGEATTENTION_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Link with SageAttention instances
target_link_libraries(${EXAMPLE_SAGEATTENTION_FWD} tile_sageattention_fwd_instances)

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

