// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kHasBiasGrad_,
          BlockSageAttentionQuantScaleEnum QScaleEnum_,
          index_t kBlockPerCu_  = -1, /* overwrite occupancy if not -1 */
          bool kSkipMinSeqlenQ_ = false /* skip min seqlen q while chunked prefill */>
struct TileSageAttnTraits
{
    static constexpr bool kPadSeqLenQ     = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK     = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ    = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV    = kPadHeadDimV_;
    static constexpr bool kHasBiasGrad    = kHasBiasGrad_;
    static constexpr auto QScaleEnum      = QScaleEnum_;
    static constexpr index_t kBlockPerCu  = kBlockPerCu_;
    static constexpr bool kSkipMinSeqlenQ = kSkipMinSeqlenQ_;
};

} // namespace ck_tile
