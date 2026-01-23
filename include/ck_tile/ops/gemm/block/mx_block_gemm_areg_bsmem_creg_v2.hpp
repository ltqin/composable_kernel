// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_default_policy.hpp"

namespace ck_tile {

// MX Block GEMM with Scale A and Scale B support
// A is block distributed tensor
// B is block window on shared memory
// C is block distributed tensor
// ScaleA and ScaleB are scale tensors for MX format (e.g., MXFP4)
template <typename Problem_, typename Policy_ = BlockGemmARegBSmemCRegV2DefaultPolicy>
struct MXBlockGemmARegBSmemCRegV2
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // MX packing parameters
    static constexpr index_t MXdlPack = 2;
    static constexpr index_t NXdlPack = 2;
    static constexpr index_t KXdlPack = 2;

    // C += A * B with Scale
    template <typename CBlockTensor,
              typename ABlockTensorTmp,
              typename BBlockWindowTmp,
              typename ScaleATensor,
              typename ScaleBTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp,
                                   const ScaleATensor& scale_a_tensor,
                                   const ScaleBTensor& scale_b_tensor) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        // MX pack iteration counts
        constexpr index_t MPackIterPerWarp = MIterPerWarp / MXdlPack;
        constexpr index_t NPackIterPerWarp = NIterPerWarp / NXdlPack;
        constexpr index_t KPackIterPerWarp = KIterPerWarp / KXdlPack;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        // construct from A-block-tensor from A-Block-tensor-tmp
        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // check C-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop with MX scale support:
        // Iterate over packed dimensions for proper scale indexing
        static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
            static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
                static_for<0, MPackIterPerWarp, 1>{}([&](auto impack) {
                    // Inner loops for XDL packing
                    static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                        static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto kIter = ikpack * KXdlPack + ikxdl;
                                constexpr auto nIter = inpack * NXdlPack + inxdl;
                                constexpr auto mIter = impack * MXdlPack + imxdl;

                                // Template parameters for WarpGemm
                                constexpr auto APackIter = ikxdl * MXdlPack + imxdl;
                                constexpr auto BPackIter = ikxdl * NXdlPack + inxdl;

                                // read B warp tensor from B Block window
                                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                                // read A warp tensor from A block tensor
                                AWarpTensor a_warp_tensor;
                                a_warp_tensor.get_thread_buffer() =
                                    a_block_tensor.get_y_sliced_thread_data(
                                        merge_sequences(sequence<mIter, kIter>{},
                                                        a_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                                // read C warp tensor from C block tensor
                                CWarpTensor c_warp_tensor;
                                c_warp_tensor.get_thread_buffer() =
                                    c_block_tensor.get_y_sliced_thread_data(
                                        merge_sequences(sequence<mIter, nIter>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                // Get scale values
                                const auto scale_a =
                                    scale_a_tensor(impack)(ikpack).get_thread_buffer()[0];
                                const auto scale_b =
                                    scale_b_tensor(inpack)(ikpack).get_thread_buffer()[0];

                                // warp GEMM with scale
                                WG{}.template operator()<APackIter, BPackIter>(
                                    c_warp_tensor, a_warp_tensor, b_warp_tensor, scale_a, scale_b);

                                // write C warp tensor into C block tensor
                                c_block_tensor.set_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                    c_warp_tensor.get_thread_buffer());
                            });
                        });
                    });
                });
            });
        });
    }

    // Overload without scale (falls back to standard GEMM)
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        // ... standard GEMM implementation (same as BlockGemmARegBSmemCRegV2)
        // This allows the same class to be used with or without scale
        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());
        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    template <index_t MPerBlock = BlockGemmShape::kM, index_t KPerBlock = BlockGemmShape::kK>
    CK_TILE_DEVICE static constexpr auto MakeABlockTileDistribution()
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        return make_static_tile_distribution(a_block_dstr_encode);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C = A * B with Scale (returns new C tensor)
    template <typename ABlockTensorTmp,
              typename BBlockWindowTmp,
              typename ScaleATensor,
              typename ScaleBTensor>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp,
                                   const ScaleATensor& scale_a_tensor,
                                   const ScaleBTensor& scale_b_tensor) const
    {
        auto c_block_tensor = MakeCBlockTile();
        clear_tile(c_block_tensor);
        operator()(
            c_block_tensor, a_block_tensor_tmp, b_block_window_tmp, scale_a_tensor, scale_b_tensor);
        return c_block_tensor;
    }

    // C = A * B without Scale (returns new C tensor)
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        clear_tile(c_block_tensor);
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
        return c_block_tensor;
    }

    // Helper to get iteration counts for external scheduler
    static constexpr index_t GetMIterPerWarp()
    {
        constexpr auto config   = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        return BlockGemmShape::kM / (MWarp * WG::kM);
    }

    static constexpr index_t GetNIterPerWarp()
    {
        constexpr auto config   = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t NWarp = config.template at<2>();
        return BlockGemmShape::kN / (NWarp * WG::kN);
    }

    static constexpr index_t GetKIterPerWarp()
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        return BlockGemmShape::kK / WG::kK;
    }

    static constexpr index_t GetMfmaCount()
    {
        return GetMIterPerWarp() * GetNIterPerWarp() * GetKIterPerWarp();
    }
};

} // namespace ck_tile
