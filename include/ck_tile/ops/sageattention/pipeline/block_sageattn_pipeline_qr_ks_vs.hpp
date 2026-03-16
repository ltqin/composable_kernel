// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"
#include "ck_tile/ops/sageattention/pipeline/block_sageattn_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
template <typename Problem_, typename Policy_ = BlockSageAttentionPipelineQRKSVSDefaultPolicy>
struct BlockSageAttentionPipelineQRKSVS
{
    using Problem             = remove_cvref_t<Problem_>;
    using Policy              = remove_cvref_t<Policy_>;
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockSageAttnShape         = remove_cvref_t<typename Problem::BlockSageAttnShape>;
    using VLayout                    = remove_cvref_t<typename BlockSageAttnShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockSageAttnShape::kM0;
    static constexpr index_t kN0           = BlockSageAttnShape::kN0;
    static constexpr index_t kK0           = BlockSageAttnShape::kK0;
    static constexpr index_t kN1           = BlockSageAttnShape::kN1;
    static constexpr index_t kK1           = BlockSageAttnShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockSageAttnShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockSageAttnShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

    static constexpr uint32_t DS_READ = 0x100; // Barrier for DS (data share) read
    static constexpr uint32_t MFMA    = 0x008; // Barrier for MFMA (matrix multiply-accumulate)

    // For BLOCKSCALE: shift value for exp2(x + shift) to scale P to [0, 2^shift]
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;
    static constexpr auto QScaleEnum      = Problem::QScaleEnum;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "qr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding /*position_encoding*/,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               [[maybe_unused]] const float* q_descale_ptr = nullptr,
               const float* k_descale_ptr                  = nullptr,
               const float* v_descale_ptr                  = nullptr,
               [[maybe_unused]] index_t block_scale_size_q = 0,
               index_t block_scale_size_k                  = 0,
               [[maybe_unused]] float q_descale_value      = 1.0f) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q = load_tile(q_dram_window);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType =
            std::conditional_t<std::is_same_v<typename SaccBlockTileType::DataType, SaccDataType>,
                               SaccBlockTileType,
                               decltype(cast_tile<SaccDataType>(SaccBlockTileType{}))>;

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        const auto q_origin = q_dram_window.get_window_origin();

        const auto tile_range_result = [&mask, &q_origin]() {
            auto [start, end] =
                mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            return ck_tile::make_tuple(start, end);
        }();
        const auto seqlen_k_start = tile_range_result.get(ck_tile::number<0>{});
        const auto seqlen_k_end   = tile_range_result.get(ck_tile::number<1>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, 0}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        // Use compile-time conditional for group barrier sequence
        // (No runtime lambda selection)
        auto schedule_gemm0 = [] {
            using BlockGemm0 = remove_cvref_t<decltype(gemm_0)>;
            constexpr auto WarpGemmConfig =
                BlockGemm0::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm0 = remove_cvref_t<decltype(WarpGemmConfig.template at<0>())>;
            constexpr index_t Gemm0MWarp   = WarpGemmConfig.template at<1>();
            constexpr index_t Gemm0NWarp   = WarpGemmConfig.template at<2>();
            constexpr index_t WarpGemm0M   = WarpGemm0::WarpGemmAttribute::Impl::kM;
            constexpr index_t WarpGemm0N   = WarpGemm0::WarpGemmAttribute::Impl::kN;
            constexpr index_t WarpGemm0K   = WarpGemm0::WarpGemmAttribute::Impl::kK;
            constexpr index_t NumMfmaInsts = (kM0 / WarpGemm0M) * (kN0 / WarpGemm0N) *
                                             (kK0 / WarpGemm0K) / (Gemm0MWarp * Gemm0NWarp);
            if constexpr(get_warp_size() == 64 && kQKHeaddim == 256)
            {
                static_assert(NumMfmaInsts % 8 == 0);
                static_for<0, NumMfmaInsts / 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(DS_READ, 2, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);    // MFMA
                    __builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);    // MFMA
                    __builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0); // DS read
                    __builtin_amdgcn_sched_group_barrier(MFMA, 4, 0);    // MFMA
                });
            }
        };

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);
        index_t thread_idx = (threadIdx.x % 64) / 32;
        // main loop
        do
        {
            float k_descale = 1.0f;
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE)
            {
                const index_t kv_idx = (seqlen_k_start + i_total_loops * kN0) / block_scale_size_k;
                k_descale            = k_descale_ptr[kv_idx];
            }
            constexpr index_t kBlockScaleK =
                (QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)     ? 64
                : (QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD) ? 16
                                                                              : 128;
            constexpr index_t kNumKScalesPW                               = kN0 / kBlockScaleK;
            constexpr index_t kNumKScalesPT                               = kN0 / kBlockScaleK / 2;
            float k_scales_perwarp[kNumKScalesPW > 0 ? kNumKScalesPW : 1] = {};
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)
            {
                const index_t kv_idx = (seqlen_k_start + i_total_loops * kN0) / block_scale_size_k;
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPW; i++)
                    k_scales_perwarp[i] = k_descale_ptr[kv_idx + i];
            }
            float k_scales_reg[kNumKScalesPT > 0 ? kNumKScalesPT : 1] = {};
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                const index_t k_global_start    = seqlen_k_start + i_total_loops * kN0;
                const index_t k_scale_start_idx = k_global_start / block_scale_size_k;
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPT; i++)
                    k_scales_reg[i] = k_descale_ptr[k_scale_start_idx + 2 * i + thread_idx];
            }

            // STAGE 1, QK gemm
            auto k_dram_window = make_tile_window(
                k_dram_block_window.get_bottom_tensor_view(),
                k_dram_block_window.get_window_lengths(),
                k_dram_block_window.get_window_origin(),
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load
            auto s_acc_gemm   = SaccBlockTileType{};
            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(k_dram_window, {0, kK0});
                clear_tile(s_acc_gemm); // initialize C
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    gemm_0(s_acc_gemm,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_lds_window);
                    schedule_gemm0();
                    block_sync_lds();
                    move_tile_window(k_dram_window, {0, kK0});

                    store_tile(
                        k_lds_window,
                        tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                    k_block_tile = load_tile(k_dram_window);                // global read i + 2
                });
            }

            const auto v_prefetch = load_tile(v_dram_window); // prefetch load v tile
            {                                                 // tail
                block_sync_lds();
                gemm_0(s_acc_gemm,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 2) * kK0>{},
                                      sequence<kM0, (k0_loops - 1) * kK0>{}),
                       k_lds_window);
                schedule_gemm0();
                block_sync_lds();

                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();

                gemm_0(s_acc_gemm,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       k_lds_window);
                schedule_gemm0();
            }

            // Convert GEMM output to SaccDataType for softmax (if needed)
            auto s_acc = [&]() {
                using GemmDataType = typename decltype(s_acc_gemm)::DataType;
                if constexpr(std::is_same_v<GemmDataType, SaccDataType>)
                {
                    return s_acc_gemm; // No conversion needed (e.g., float -> float)
                }
                else
                {
                    return cast_tile<SaccDataType>(s_acc_gemm); // Convert (e.g., int32 -> float)
                }
            }();

            // dequant: combine q_descale (in s_acc_element_func) with k_descale
            auto s_acc_element_func_ = [&]() {
                if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE)
                {
                    return s_acc_element_func * k_descale;
                }
                else
                    return s_acc_element_func;
            }();

            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                float combined_scales_reg[kNumKScalesPT] = {};
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPT; i++)
                    combined_scales_reg[i] = q_descale_value * k_scales_reg[i];
                int count                  = 0;
                constexpr auto s_acc_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_acc_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_acc_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= combined_scales_reg[count >> 4];
                        count++;
                    });
                });
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)
            {
                float combined_scales_reg[kNumKScalesPW] = {};
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPW; i++)
                    combined_scales_reg[i] = q_descale_value * k_scales_perwarp[i];
                int count                  = 0;
                constexpr auto s_acc_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_acc_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_acc_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= combined_scales_reg[count >> 5];
                        count++;
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func_, s_acc);
            }
            // STAGE 2, scale_s, mask, softmax
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    auto apply_mask = [&](auto&& mask_func) {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col =
                                    k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                return !mask_func(variant_params,
                                                  block_indices.batch_idx,
                                                  row,
                                                  col,
                                                  block_indices.qo_head_idx,
                                                  block_indices.kv_head_idx);
                            });
                    };

                    apply_mask([&](auto&&... args) {
                        return variant.LogitsMask(std::forward<decltype(args)>(args)...);
                    });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                if constexpr(FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // For BLOCKSCALE: precompute (m - shift) once per row
                // Bias/Alibi: exp2(s - m + shift) = exp2(s - (m - shift))
                // else: exp2(scale_s*s - scale_s*m + shift) = exp2(scale_s*s - (scale_s*m - shift))
                auto validated_m = get_validated_m(m[i_idx]);
                auto row_max     = scale_s * validated_m;
                if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                             QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                             QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
                {
#if CK_TILE_USE_OCP_FP8
                    validated_m -= OCP_FP8_SHIFT; // for Bias/Alibi
                    row_max -= OCP_FP8_SHIFT;     // for else branch
#else
                    validated_m -= FNUZ_FP8_SHIFT;
                    row_max -= FNUZ_FP8_SHIFT;
#endif
                }
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx)     = exp2(scale_s * s[i_j_idx] - row_max);
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_new = get_validated_m(m[i_idx]);
                auto row_max     = scale_s * m_new;
                const auto tmp   = exp2(scale_s * m_old[i_idx] - row_max);
                // Update l and rescale o_acc
                l(i_idx) = tmp * l(i_idx) + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds();
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_prefetch);
                store_tile(
                    v_lds_window,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_window,
                           tile_elementwise_in(v_element_func, v_prefetch)); // store the prefetch
            }
            move_tile_window(v_dram_window, {0, kK1});

            const auto p =
                cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));

            // STAGE 3, KV gemm
            // For BLOCKSCALE, PERWARP, and PERTHREAD modes, accumulate directly to o_acc
            // Apply per-channel v_descale after the loop (before normalization)

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_lds_window);
                    block_sync_lds();
                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v);
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func, v)); // store next v
                    }
                    move_tile_window(v_dram_window, {0, kK1});
                });
            }
            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});
            // tail
            {
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                       v_lds_window);
                block_sync_lds();
            }

        } while(++i_total_loops < num_total_loop);

        // Apply per-channel v_descale for BLOCKSCALE, PERWARP, and PERTHREAD modes (after loop,
        // before normalization)
        if constexpr(Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                     Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::PERWARP ||
                     Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::PERTHREAD)
        {
            // V is col-major, each column (channel) has its own scale
            // o_acc shape: [M0, N1] where N1 is hdim_v
            // v_descale_ptr points to per-channel scales [hdim_v]
            // Load v_descale to LDS for better memory access pattern
            // Reuse K/V LDS space (they're no longer needed)
            auto v_descale_lds = reinterpret_cast<float*>(smem_ptr);

            // Cooperatively load v_descale to LDS
            const index_t num_threads = kBlockSize;
            for(index_t i = threadIdx.x; i < kN1; i += num_threads)
            {
                v_descale_lds[i] = v_descale_ptr[i];
            }
            block_sync_lds();

            constexpr auto o_tmp_spans = decltype(o_acc)::get_distributed_spans();

            sweep_tile_span(o_tmp_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(o_tmp_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // Get the global tile index for the N1 (channel) dimension
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        o_acc.get_tile_distribution(), i_j_idx);
                    const index_t channel_idx = tile_idx.at(number<1>{});
                    const float v_scale       = v_descale_lds[channel_idx];
                    o_acc(i_j_idx) *= v_scale;
                });
            });
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                // When masking, the denominator can be zero; guard the normalization
                // so we do not divide by zero after a fully masked row.
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               [[maybe_unused]] const float* q_descale_ptr = nullptr,
               const float* k_descale_ptr                  = nullptr,
               const float* v_descale_ptr                  = nullptr,
               [[maybe_unused]] index_t block_scale_size_q = 0,
               index_t block_scale_size_k                  = 0,
               [[maybe_unused]] float q_descale_value      = 1.0f) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          q_descale_ptr,
                          k_descale_ptr,
                          v_descale_ptr,
                          block_scale_size_q,
                          block_scale_size_k,
                          q_descale_value);
    }
};

} // namespace ck_tile
