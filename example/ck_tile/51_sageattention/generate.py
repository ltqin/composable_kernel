# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
import pkgutil
from typing import List, Optional

import codegen.ops
from codegen.cmake_config import GEN_DIR


class HandlerId(IntEnum):
    LIST_BLOBS = 0
    WRITE_BLOBS = 1


# inspect all modules under 'codegen.ops' and register API handlers
ops = []
for importer, module_name, _ in pkgutil.iter_modules(codegen.ops.__path__):
    full_module_name = "%s.%s" % (codegen.ops.__name__, module_name)
    ops.append(importer.find_spec(module_name).loader.load_module(module_name))
unwanted_prefix = "sageattention_"
handlers = dict(
    [
        (
            op.__name__[len(unwanted_prefix) :]
            if op.__name__.startswith(unwanted_prefix)
            else op.__name__,
            (op.list_blobs, op.write_blobs),
        )
        for op in ops
    ]
)
assert 0 < len(handlers)


def write_blobs(
    targets: List[str],
    output_dir: Optional[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
) -> None:
    if output_dir is None:
        output_dir = GEN_DIR

    for api in api_list:
        if api not in handlers:
            raise ValueError(
                f"Unknown API: {api}. Available APIs: {list(handlers.keys())}"
            )

        handler = handlers[api][HandlerId.WRITE_BLOBS]
        handler(targets, output_dir, optdim_list, filters_list)


def list_blobs(
    targets: List[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
    list_blobs_file: Optional[str],
) -> None:
    all_blobs = []

    for api in api_list:
        if api not in handlers:
            raise ValueError(
                f"Unknown API: {api}. Available APIs: {list(handlers.keys())}"
            )

        handler = handlers[api][HandlerId.LIST_BLOBS]
        blobs = handler(targets, optdim_list, filters_list)
        all_blobs.extend(blobs)

    if list_blobs_file:
        with open(list_blobs_file, "w") as f:
            for blob in all_blobs:
                f.write(f"{blob}\n")
    else:
        for blob in all_blobs:
            print(blob)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SageAttention kernel instances"
    )

    parser.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Comma-separated list of GPU targets (e.g., gfx90a,gfx942,gfx1100)",
    )

    parser.add_argument(
        "--api",
        type=str,
        default="fwd",
        help="Comma-separated list of APIs to generate (e.g., fwd)",
    )

    parser.add_argument(
        "--optdim",
        type=str,
        default="64,128",
        help="Comma-separated list of head dimensions to optimize for (e.g., 32,64,128,256)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Comma-separated list of filters (not implemented yet)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated files",
    )

    parser.add_argument(
        "--list_blobs",
        type=str,
        default=None,
        help="Output file to list all blob filenames (instead of generating them)",
    )

    args = parser.parse_args()

    # Parse arguments
    targets = [t.strip() for t in args.targets.split(",")]
    api_list = [a.strip() for a in args.api.split(",")]
    optdim_list = [int(d.strip()) for d in args.optdim.split(",")]
    filters_list = [f.strip() for f in args.filter.split(",")] if args.filter else []

    if args.list_blobs:
        list_blobs(targets, api_list, filters_list, optdim_list, args.list_blobs)
    else:
        write_blobs(targets, args.output_dir, api_list, filters_list, optdim_list)


if __name__ == "__main__":
    main()
