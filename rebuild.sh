#!/bin/bash

# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
rm build -rf
mkdir -p build
cd build

../script/cmake-ck-dev.sh  ../ gfx942 -G Ninja -D FMHA_FWD_ENABLE_APIS="fwd"
#ninja  tile_example_fmha_fwd
ninja  tile_example_sageattention_fwd
# ninja  tile_test_vsa_sparse_attn
#-D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++                                                  \
#-D CMAKE_CXX_COMPILER=/mydata/sourcecode/z-llvm-project/build/bin/clang++                                                  \

# GPU_TARGETS=gfx942

# cmake                                                                                             \
# -D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
# -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++                                                  \
# -D CMAKE_CXX_FLAGS=" -g -O0   -Wno-gnu-line-marker"     \
# -D CMAKE_BUILD_TYPE=Release                                                                      \
# -D BUILD_DEV=ON                                                                                   \
# -D GPU_TARGETS=$GPU_TARGETS                                                                       \
# ..

# make -j    tile_example_fmha_fwd


