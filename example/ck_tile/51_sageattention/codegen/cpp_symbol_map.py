# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
FWD_DTYPE_MAP = {
    "fp32": "SageAttentionFwdFp32",
    "fp16": "SageAttentionFwdFp16",
    "bf16": "SageAttentionFwdBf16",
    "fp8": "SageAttentionFwdFp8",
    "fp8fp16": "SageAttentionFwdFp8Fp16",
    "fp8bf16": "SageAttentionFwdFp8Bf16",
    "fp8fp32": "SageAttentionFwdFp8Fp32",
}

BWD_DTYPE_MAP = {"fp32": "FmhaBwdFp32", "fp16": "FmhaBwdFp16", "bf16": "FmhaBwdBf16"}

MASK_IMPL = {
    "generic": "ck_tile::GenericAttentionMask",
    "simplified": "ck_tile::SimplifiedGenericAttentionMask",
}

_MASK_SIMPLIFIED_MAP = {
    "s_no": "ck_tile::SimplifiedGenericAttentionMask<false>",
    "s_mask": "ck_tile::SimplifiedGenericAttentionMask<true>",
}

_MASK_MAP = {
    "no": "FmhaMasks::NoMask",
    "causal": "FmhaMasks::CausalMask",
    "generic": "FmhaMasks::GenericMask",
}


def get_mask_map(mask_impl: str):
    if mask_impl == "generic":
        return _MASK_MAP
    elif mask_impl == "simplified":
        return _MASK_SIMPLIFIED_MAP
    else:
        assert False
        return None


def get_mask_impl(mask: str) -> str:
    return "simplified" if mask.startswith("s_") else "generic"


def get_mask_cpp_type(mask: str) -> str:
    return get_mask_map(get_mask_impl(mask))[mask]


_MASK_CHECK_MAP = {
    "no": "t.mask_type == mask_enum::no_mask",
    "causal": "t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right",
    "generic": "t.mask_type == mask_enum::window_generic",
}

_MASK_SIMPLIFIED_CHECK_MAP = {
    "s_no": "t.mask_type == mask_enum::no_mask",
    "s_mask": "t.mask_type != mask_enum::no_mask",
}


def get_mask_check_map(mask: str):
    if mask == "generic":
        return _MASK_CHECK_MAP
    elif mask == "simplified":
        return _MASK_SIMPLIFIED_CHECK_MAP
    else:
        assert False
        return None


def get_mask_cpp_check_expr(mask: str) -> str:
    return get_mask_check_map(get_mask_impl(mask))[mask]


QSCALE_MAP = {
    "no": "ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE",
    "pertensor": "ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR",
}

QSCALE_CHECK_MAP = {
    "no": "quant_scale_enum::no_scale",
    "pertensor": "quant_scale_enum::pertensor",
}

BIAS_MAP = {
    "no": "ck_tile::BlockAttentionBiasEnum::NO_BIAS",
    "bias": "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS",
    "alibi": "ck_tile::BlockAttentionBiasEnum::ALIBI",
}

# TODO: this is ugly
BIAS_CHECK_MAP = {
    "no": "bias_enum::no_bias",
    "bias": "bias_enum::elementwise_bias",
    "alibi": "bias_enum::alibi",
}

DROPOUT_MAP = {
    "no": "ck_tile::BlockDropoutBwd<false, true,  false>",
    "dropout_wg32": "ck_tile::BlockDropoutBwd<true,  true,  false>",
    "dropout_wg32_storerandval": "ck_tile::BlockDropoutBwd<true,  true,  true >",
    "dropout_wg16": "ck_tile::BlockDropoutBwd<true,  false, false>",
    "dropout_wg16_storerandval": "ck_tile::BlockDropoutBwd<true,  false, true >",
}

DROPOUT_CHECK_MAP = {
    "no": "t.has_dropout == false",
    "dropout_wg32": "t.has_dropout == true && t.is_store_randval == false",
    "dropout_wg32_storerandval": "t.has_dropout == true && t.is_store_randval == true",
    "dropout_wg16": "t.has_dropout == true && t.is_store_randval == false",
    "dropout_wg16_storerandval": "t.has_dropout == true && t.is_store_randval == true",
}

ROPE_MAP = {
    "no": "ck_tile::RotaryEmbeddingEnum::NONE",
    "inter": "ck_tile::RotaryEmbeddingEnum::INTERLEAVED",
    "half": "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED",
}

ROPE_CHECK_MAP = {
    "no": "rope_enum::none",
    "inter": "rope_enum::interleaved",
    "half": "rope_enum::half_rotated",
}

MODE_MAP = {"batch": "false", "group": "true"}

LAYOUT_MAP = {"row": "true", "col": "false"}

PIPELINE_MAP = {
    "qr": "ck_tile::BlockSageAttentionPipelineQRKSVS",
    "qr_async": "ck_tile::BlockSageAttentionPipelineQRKSVSAsync",
}

PIPELINE_ENUM_MAP = {
    "qr": "ck_tile::BlockSageAttnPipelineEnum::QRKSVS",
    "qr_async": "ck_tile::BlockSageAttnPipelineEnum::QRKSVS_ASYNC",
}

BOOL_MAP = {
    "t": "true",
    "f": "false",
    True: "true",
    False: "false",
}
