// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockSageAttnPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QSKSVS,
    QRKSVS_ASYNC_TRLOAD,
    QRKSVS_ASYNC_TRLOAD_V3,
};

template <BlockSageAttnPipelineEnum>
struct BlockSageAttnPipelineEnumToStr;

template <>
struct BlockSageAttnPipelineEnumToStr<BlockSageAttnPipelineEnum::QRKSVS>
{
    static constexpr const char* name = "qr";
};
template <>
struct BlockSageAttnPipelineEnumToStr<BlockSageAttnPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};

} // namespace ck_tile
