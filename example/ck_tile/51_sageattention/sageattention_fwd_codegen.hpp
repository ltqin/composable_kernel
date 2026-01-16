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

// Data type tags for SageAttention
struct SageAttnFwdFp32
{
};
struct SageAttnFwdFp16
{
};
struct SageAttnFwdBf16
{
};
struct SageAttnFwdFp8
{
};
struct SageAttnFwdFp8Fp16
{
};
struct SageAttnFwdFp8Bf16
{
};
struct SageAttnFwdFp8Fp32
{
};

// Type configuration for SageAttention
template <typename DataType>
struct SageAttentionFwdTypeConfig;

template <>
struct SageAttentionFwdTypeConfig<SageAttnFwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using PDataType             = ck_tile::half_t;
    using OaccDataType          = float;
    using ODataType             = ck_tile::half_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttnFwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using PDataType             = ck_tile::bf16_t;
    using OaccDataType          = float;
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttnFwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using PDataType             = float;
    using OaccDataType          = float;
    using ODataType             = float;
};

// Mask definitions for SageAttention
namespace SageAttnMasks {
using NoMask      = ck_tile::SimplifiedGenericAttentionMask<false>;
using CausalMask  = ck_tile::SimplifiedGenericAttentionMask<true>;
using GenericMask = ck_tile::GenericAttentionMask;
} // namespace SageAttnMasks

// Trait structure for kernel selection
template <int kHdim,
          typename DataType,
          bool kIsGroupMode,
          int kM0,
          int kN0,
          int kK0,
          int kN1,
          int kK1,
          int kK0BlockMax,
          bool kIsVLayoutRowMajor,
          ck_tile::BlockFmhaPipelineEnum PipelineEnum,
          bool kHasLogitsSoftCap,
          typename MaskType,
          ck_tile::BlockAttentionBiasEnum BiasEnum,
          bool kStoreLSE,
          bool kHasDropout,
          ck_tile::BlockAttentionQuantScaleEnum QScaleEnum,
          bool kPadSeqLenQ,
          bool kPadSeqLenK,
          bool kPadHeadDimQ,
          bool kPadHeadDimV,
          bool kTransposeLoad,
          bool kSkipMinSeqlenQ,
          bool kHasSink>
struct sageattention_fwd_traits_
{
    static constexpr int hdim                                               = kHdim;
    static constexpr bool is_group_mode                                     = kIsGroupMode;
    static constexpr int kM0_                                               = kM0;
    static constexpr int kN0_                                               = kN0;
    static constexpr int kK0_                                               = kK0;
    static constexpr int kN1_                                               = kN1;
    static constexpr int kK1_                                               = kK1;
    static constexpr int kK0BlockMax_                                       = kK0BlockMax;
    static constexpr bool is_v_rowmajor                                     = kIsVLayoutRowMajor;
    static constexpr ck_tile::BlockFmhaPipelineEnum pipeline_enum           = PipelineEnum;
    static constexpr bool has_logits_soft_cap                               = kHasLogitsSoftCap;
    static constexpr ck_tile::BlockAttentionBiasEnum bias_enum              = BiasEnum;
    static constexpr bool store_lse                                         = kStoreLSE;
    static constexpr bool has_dropout                                       = kHasDropout;
    static constexpr ck_tile::BlockAttentionQuantScaleEnum quant_scale_enum = QScaleEnum;
    static constexpr bool pad_seqlen_q                                      = kPadSeqLenQ;
    static constexpr bool pad_seqlen_k                                      = kPadSeqLenK;
    static constexpr bool pad_headdim_q                                     = kPadHeadDimQ;
    static constexpr bool pad_headdim_v                                     = kPadHeadDimV;
    static constexpr bool transpose_load                                    = kTransposeLoad;
    static constexpr bool skip_min_seqlen_q                                 = kSkipMinSeqlenQ;
    static constexpr bool has_sink                                          = kHasSink;

    using data_type = DataType;
    using mask_type = MaskType;
};

// Forward declarations for kernel instances
struct sageattention_fwd_args;

template <typename Trait, typename ArchTag>
float sageattention_fwd_(const ck_tile::stream_config& s, sageattention_fwd_args a);

// Helper function to create kernel arguments
template <typename Kernel>
auto make_sageattention_fwd_kargs(const sageattention_fwd_args& args);

#endif // SAGEATTENTION_FWD_CODEGEN_HPP
