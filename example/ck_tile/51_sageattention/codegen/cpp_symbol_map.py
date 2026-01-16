# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation

# SageAttention specific data type mappings
FWD_DTYPE_MAP = {
    "fp32": "SageAttnFwdFp32",
    "fp16": "SageAttnFwdFp16",
    "bf16": "SageAttnFwdBf16",
    "fp8": "SageAttnFwdFp8",
    "fp8fp16": "SageAttnFwdFp8Fp16",
    "fp8bf16": "SageAttnFwdFp8Bf16",
    "fp8fp32": "SageAttnFwdFp8Fp32",
}

MASK_IMPL = {
    "generic": "ck_tile::GenericAttentionMask",
    "simplified": "ck_tile::SimplifiedGenericAttentionMask",
}

_MASK_SIMPLIFIED_MAP = {
    "s_no": "ck_tile::SimplifiedGenericAttentionMask<false>",
    "s_mask": "ck_tile::SimplifiedGenericAttentionMask<true>",
}

_MASK_MAP = {
    "no": "SageAttnMasks::NoMask",
    "causal": "SageAttnMasks::CausalMask",
    "generic": "SageAttnMasks::GenericMask",
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

BIAS_CHECK_MAP = {
    "no": "bias_enum::no_bias",
    "bias": "bias_enum::elementwise_bias",
    "alibi": "bias_enum::alibi",
}

MODE_MAP = {"batch": "false", "group": "true"}

LAYOUT_MAP = {"row": "true", "col": "false"}

# SageAttention uses qr_async pipeline
PIPELINE_MAP = {
    "qr_async": "ck_tile::BlockFmhaPipelineQRKSVSAsync",
}

PIPELINE_ENUM_MAP = {
    "qr_async": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
}

BOOL_MAP = {
    "t": "true",
    "f": "false",
    True: "true",
    False: "false",
}
