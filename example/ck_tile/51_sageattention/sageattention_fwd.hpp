// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/sageattention.hpp"

#include "bias.hpp"
#include "mask.hpp"
#include "quant.hpp"
#include "rotary.hpp"

#include <type_traits>
#include <utility>
#include <variant>

// SageAttention data type configs
struct SageAttentionFwdFp32
{
};

struct SageAttentionFwdFp16
{
};

struct SageAttentionFwdBf16
{
};

struct SageAttentionFwdFp8
{
};

struct SageAttentionFwdBf8
{
};

struct SageAttentionFwdFp8Fp16
{
};

struct SageAttentionFwdFp8Bf16
{
};

struct SageAttentionFwdFp8Fp32
{
};

template <typename DataType>
struct SageAttentionFwdTypeConfig;

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = float; // data type for A matrix of second gemm
    using OaccDataType          = float; // data type for second gemm accumulation
    using ODataType             = float;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp8>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdBf8>
{
    using QDataType             = ck_tile::bf8_t;
    using KDataType             = ck_tile::bf8_t;
    using VDataType             = ck_tile::bf8_t;
    using BiasDataType          = ck_tile::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp8Bf16>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp8Fp32>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = float;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct sageattn_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //     lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //     sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //     exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

struct fmha_batch_prefill_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode (kvcache):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.page_block_size * (kargs.kv_indptr[b + 1] - kargs.kv_indptr[b] -
    //             1) +
    //                        kargs.kv_last_page_lens[b]
    // group mode (kvcache):
    //             seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.page_block_size * (kargs.kv_indptr[b + 1] - kargs.kv_indptr[b] -
    //             1) +
    //                        kargs.kv_last_page_lens[b]
    const void* seqstart_q_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    // KV cache page table fields (kv_lookup_table selects interpretation):
    // - SGLANG_PAGE_TABLE_1D:
    //   kv_indptr: prefix-sum [batch+1] into kv_page_indices
    //   kv_page_indices: 1D list of physical page ids, length = num_total_pages
    //   kv_last_page_lens: per-batch last page lengths [batch]
    // - VLLM_BLOCK_TABLE_2D:
    //   kv_page_indices: block_table [batch, max_blocks_per_seq] (2D)
    //   batch_stride_block_table: row stride for block_table
    //   seqlen_k_ptr: per-batch seqlen_k [batch]
    int32_t num_total_pages;          // total physical pages in KV cache (SGLang/vLLM)
    ck_tile::index_t page_block_size; // tokens per page (SGLang/vLLM)
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum
        kv_memory_layout;                                          // KV memory layout (SGLang/vLLM)
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table; // lookup table layout selector
    void* kv_indptr;                           // SGLang: prefix-sum; vLLM: unused
    void* kv_page_indices;                     // SGLang: 1D page list; vLLM: block_table 2D
    void* kv_last_page_lens;                   // SGLang: last page lengths; vLLM: unused
    void* seqlen_k_ptr;                        // vLLM: per-batch seqlen_k; SGLang: unused
    ck_tile::index_t batch_stride_block_table; // vLLM: row stride; SGLang: unused

    float scale_s;
    float scale_p;
    float scale_o;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

template <typename SageAttnKernel>
auto sageattn_fwd_create_kargs_and_grids(sageattn_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(SageAttnKernel::kIsGroupMode)
        {
            return SageAttnKernel::MakeKargsImpl(args.q_ptr,
                                                 args.k_ptr,
                                                 args.v_ptr,
                                                 args.bias_ptr,
                                                 args.q_descale_ptr,
                                                 args.k_descale_ptr,
                                                 args.v_descale_ptr,
                                                 args.rand_val_ptr,
                                                 args.lse_ptr,
                                                 args.o_ptr,
                                                 args.seqstart_q_ptr,
                                                 args.seqstart_k_ptr,
                                                 args.seqlen_q_ptr,
                                                 args.seqlen_k_ptr,
                                                 args.hdim_q,
                                                 args.hdim_v,
                                                 args.nhead_q,
                                                 args.nhead_q / args.nhead_k,
                                                 args.scale_s,
                                                 args.stride_q,
                                                 args.stride_k,
                                                 args.stride_v,
                                                 args.stride_bias,
                                                 args.stride_randval,
                                                 args.stride_o,
                                                 args.nhead_stride_q,
                                                 args.nhead_stride_k,
                                                 args.nhead_stride_v,
                                                 args.nhead_stride_bias,
                                                 args.nhead_stride_randval,
                                                 args.nhead_stride_lse,
                                                 args.nhead_stride_o,
                                                 args.window_size_left,
                                                 args.window_size_right,
                                                 args.sink_size,
                                                 args.mask_type,
                                                 args.min_seqlen_q,
                                                 args.p_drop,
                                                 args.s_randval,
                                                 args.drop_seed_offset,
                                                 args.cu_seqlen_q_ptr,
                                                 args.cu_seqlen_k_ptr,
                                                 args.sink_ptr);
        }
        else
        { // create batch mode kernel arguments
            return SageAttnKernel::MakeKargsImpl(args.q_ptr,
                                                 args.k_ptr,
                                                 args.v_ptr,
                                                 args.bias_ptr,
                                                 args.q_descale_ptr,
                                                 args.k_descale_ptr,
                                                 args.v_descale_ptr,
                                                 args.rand_val_ptr,
                                                 args.lse_ptr,
                                                 args.o_ptr,
                                                 args.seqlen_q,
                                                 args.seqlen_k,
                                                 args.hdim_q,
                                                 args.hdim_v,
                                                 args.nhead_q,
                                                 args.nhead_q / args.nhead_k,
                                                 args.scale_s,
                                                 args.stride_q,
                                                 args.stride_k,
                                                 args.stride_v,
                                                 args.stride_bias,
                                                 args.stride_randval,
                                                 args.stride_o,
                                                 args.nhead_stride_q,
                                                 args.nhead_stride_k,
                                                 args.nhead_stride_v,
                                                 args.nhead_stride_bias,
                                                 args.nhead_stride_randval,
                                                 args.nhead_stride_lse,
                                                 args.nhead_stride_o,
                                                 args.batch_stride_q,
                                                 args.batch_stride_k,
                                                 args.batch_stride_v,
                                                 args.batch_stride_bias,
                                                 args.batch_stride_randval,
                                                 args.batch_stride_lse,
                                                 args.batch_stride_o,
                                                 args.window_size_left,
                                                 args.window_size_right,
                                                 args.sink_size,
                                                 args.mask_type,
                                                 args.p_drop,
                                                 args.s_randval,
                                                 args.drop_seed_offset,
                                                 args.cu_seqlen_q_ptr,
                                                 args.cu_seqlen_k_ptr,
                                                 args.sink_ptr);
        }
    }();

    if constexpr(SageAttnKernel::kIsGroupMode)
    {
        dim3 grids = SageAttnKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids = SageAttnKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

template <typename SageAttnKernel>
auto sageattn_fwd_v3_create_kargs_and_grids(sageattn_fwd_args args)
{
    /// NOTICE: This was borrowed from Aiter. Make sure the selected remap_opt setting truly
    /// maximizes the kernel's performance.
    int remap_opt = 2;
    if(args.mask_type != static_cast<int>(mask_enum::no_mask) &&
       ((args.nhead_q % 8 != 0) || (16384 < args.seqlen_q)))
    {
        if(65536 <= args.seqlen_q)
        {
            remap_opt = 0;
        }
        else
        {
            remap_opt = 1;
        }
    }

    auto kargs = [&] {
        if constexpr(SageAttnKernel::kIsGroupMode)
        {
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             nullptr, // lse_ptr
                                             args.o_ptr,
                                             args.seqstart_q_ptr,
                                             args.seqstart_k_ptr,
                                             args.seqlen_q_ptr,
                                             args.seqlen_k_ptr,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.scale_s,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             0, // nhead_stride_lse
                                             args.nhead_stride_o,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             remap_opt,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr);
        }
        else
        {
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             nullptr, // lse_ptr
                                             args.o_ptr,
                                             args.seqlen_q,
                                             args.seqlen_k,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.scale_s,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             0, // nhead_stride_lse
                                             args.nhead_stride_o,
                                             args.batch_stride_q,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             0, // batch_stride_lse
                                             args.batch_stride_o,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             remap_opt,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr);
        }
    }();

    dim3 grids = SageAttnKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);

    return ck_tile::make_tuple(kargs, grids);
}

template <typename SageAttnKernel>
auto sageattn_batch_prefill_create_kargs_and_grids(fmha_batch_prefill_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    using PageTableKargs            = typename SageAttnKernel::PageBlockTableKargs;
    const PageTableKargs page_table = [&]() {
        if constexpr(SageAttnKernel::kKVLookupTable ==
                     ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
        {
            return PageTableKargs{reinterpret_cast<const int32_t*>(args.kv_indptr),
                                  reinterpret_cast<const int32_t*>(args.kv_page_indices),
                                  reinterpret_cast<const int32_t*>(args.kv_last_page_lens)};
        }
        else
        {
            return PageTableKargs{reinterpret_cast<const int32_t*>(args.kv_page_indices),
                                  args.batch_stride_block_table,
                                  reinterpret_cast<const int32_t*>(args.seqlen_k_ptr)};
        }
    }();
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(SageAttnKernel::kIsGroupMode)
        {
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.bias_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqstart_q_ptr,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.num_total_pages,
                                             args.page_block_size,
                                             page_table,
                                             args.scale_s,
                                             args.scale_p,
                                             args.scale_o,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.sink_size,
                                             args.mask_type,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset,
                                             args.sink_ptr);
        }
        else
        { // create batch mode kernel arguments
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.bias_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqlen_q,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.num_total_pages,
                                             args.page_block_size,
                                             page_table,
                                             args.scale_s,
                                             args.scale_p,
                                             args.scale_o,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.batch_stride_q,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             args.batch_stride_bias,
                                             args.batch_stride_randval,
                                             args.batch_stride_lse,
                                             args.batch_stride_o,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.sink_size,
                                             args.mask_type,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset,
                                             args.sink_ptr);
        }
    }();

    dim3 grids = SageAttnKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockSageAttnPipelineEnum FmhaPipelineEnum_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          ck_tile::BlockAttentionQuantScaleEnum QScaleEnum_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_ = false,
          bool kHasSink_        = false>
struct sageattn_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = false; // always disabled for sageattention
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kHasDropout                = kHasDropout_;
    static constexpr auto QScaleEnum                 = QScaleEnum_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
    static constexpr bool kHasSink                   = kHasSink_;
};

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockSageAttnPipelineEnum FmhaPipelineEnum_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          ck_tile::BlockAttentionQuantScaleEnum QScaleEnum_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_            = false,
          ck_tile::index_t kPageBlockSize_ = 1,
          ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kKVMemoryLayout_ =
              ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT,
          ck_tile::BlockAttentionKVCacheLookupTableEnum kKVLookupTable_ =
              ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D>
struct fmha_fwd_batch_prefill_traits_ : public sageattn_fwd_traits_<HDim_,
                                                                    DataType_,
                                                                    kIsGroupMode_,
                                                                    kM0_,
                                                                    kN0_,
                                                                    kK0_,
                                                                    kN1_,
                                                                    kK1_,
                                                                    kK0BlockLength_,
                                                                    kIsVLayoutRowMajor_,
                                                                    FmhaPipelineEnum_,
                                                                    FmhaMask_,
                                                                    BiasEnum_,
                                                                    kStoreLse_,
                                                                    kHasDropout_,
                                                                    QScaleEnum_,
                                                                    kPadS_,
                                                                    kPadSK_,
                                                                    kPadD_,
                                                                    kPadDv_,
                                                                    kUseTrLoad_,
                                                                    kSkipMinSeqlenQ_,
                                                                    false>
{
    static constexpr auto kKVMemoryLayout            = kKVMemoryLayout_;
    static constexpr auto kKVLookupTable             = kKVLookupTable_;
    static constexpr ck_tile::index_t kPageBlockSize = kPageBlockSize_;
    static_assert(kIsVLayoutRowMajor_, "Batch prefill only supports row-major V layout");
};

template <typename Traits_, typename Arch = void>
float sageattn_fwd_(const ck_tile::stream_config&, sageattn_fwd_args);

// This is the public API, will be generated by script
struct sageattn_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool has_dropout;
    quant_scale_enum qscale_type;
    bool skip_min_seqlen_q = false;
    bool has_sink          = false;
    // TODO: padding check is inside this api
};
float sageattn_fwd(sageattn_fwd_traits, sageattn_fwd_args, const ck_tile::stream_config&);

struct fmha_batch_prefill_traits : public sageattn_fwd_traits
{
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    int page_size = 1;
};

float fmha_batch_prefill(fmha_batch_prefill_traits,
                         fmha_batch_prefill_args,
                         const ck_tile::stream_config&);
