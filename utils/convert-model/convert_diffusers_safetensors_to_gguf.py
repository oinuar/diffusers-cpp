#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType, quantize


def map_tensor_name(name):
    """
    Translate Diffusers tensor names into the tensor names expected by GGUF.

    Returning None skips a tensor.
    """
    return name


def map_tensor_quantization(name, dtype, qtype, dtype_to_qtype={
    #torch.bfloat16: GGMLQuantizationType.BF16,
    torch.bfloat16: GGMLQuantizationType.F32, # TODO: fix
    torch.float64: GGMLQuantizationType.F64,
    torch.float32: GGMLQuantizationType.F32,
    torch.float16: GGMLQuantizationType.F16,
    torch.int8: GGMLQuantizationType.I8,
    torch.int16: GGMLQuantizationType.I16,
    torch.int32: GGMLQuantizationType.I32,
    torch.int64: GGMLQuantizationType.I64
}):
    if qtype:
        return GGMLQuantizationType[qtype]
    
    return dtype_to_qtype[dtype]


def normalize_tensor(tensor):
    """
    Normalizes torch tensor for GGUF writing.
    """
    tensor = tensor.detach().cpu().contiguous()

    # Numpy cannot represent BF16, convert to float32
    # TODO: fix
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    # Convert to numpy
    tensor = tensor.numpy()

    return tensor


def quantize_tensor(name, tensor, qtype):
    """
    Force tensor quantization.
    """

    # Use tensor as-is
    if qtype in (GGMLQuantizationType.F64, GGMLQuantizationType.F32, GGMLQuantizationType.I64):
        return tensor

    # Downcast to float16
    if qtype == GGMLQuantizationType.F16:
        return tensor.astype(np.float16)

    # Everything below MUST be quantized manually
    return quantize(tensor, qtype)


def add_diffusion_metadata(writer, quant_type):
    """
    Add generic metadata.

    Replace/add keys based on your target runtime's documented GGUF schema.
    """
    writer.add_string("general.file_type", quant_type)
    writer.add_string("general.source.format", "safetensors")
    writer.add_string("general.source.layout", "diffusers")


def load_safetensors_model(path, tensor_names=None):
    print(f"LOAD {path}")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        tensor_names = tensor_names if tensor_names else f.keys()

        for name in tensor_names:
            tensor = f.get_tensor(name)
            yield (name, tensor)


def load_hf_model(path):
    """
    Loads ALL tensors from a Diffusers sharded checkpoint using index file.
    """

    index_file = path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Missing index file: {index_file}\n"
            "This converter requires Diffusers shard index.json"
        )

    with open(index_file, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # shard_file -> tensor names
    shard_map = {}
    for tensor_name, shard_file in weight_map.items():
        shard_map.setdefault(shard_file, []).append(tensor_name)

    for shard_file, tensor_names in shard_map.items():
        shard_path = path / shard_file
        yield from load_safetensors_model(shard_path, tensor_names)


def load_model(path):
    if path.is_dir():
        yield from load_hf_model(path)
    else:
        yield from load_safetensors_model(path)


def convert(input_path: Path, output_path: Path, architecture: str, quant_type: str) -> None:
    writer = GGUFWriter(
        path=str(output_path),
        arch=architecture,
    )

    add_diffusion_metadata(writer, quant_type)

    converted = 0
    skipped = 0

    for name, tensor in load_model(input_path):
        target_name = map_tensor_name(name)

        if target_name is None:
            skipped += 1
            continue

        source_type = tensor.dtype
        dest_type = map_tensor_quantization(target_name, source_type, quant_type)

        tensor = normalize_tensor(tensor)
        tensor = quantize_tensor(target_name, tensor, dest_type)

        writer.add_tensor(
            name=target_name,
            tensor=tensor,
            raw_dtype=dest_type,
        )

        converted += 1

        print(f"CONVERT {name} ({source_type}) -> {target_name} ({dest_type.name}) {tensor.shape}")

    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done: {output_path}")
    print(f"Converted tensors: {converted}")
    print(f"Skipped tensors:   {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Diffusers safetensors file into a GGUF file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input Diffusers folder",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .gguf file",
    )
    parser.add_argument(
        "--architecture",
        default="stable-diffusion",
        help="Target GGUF architecture/schema",
    )
    parser.add_argument(
        "--type",
        default=None,
        choices=['BF16', 'F16', 'F32', 'F64', 'I16', 'I32', 'I64', 'I8', 'IQ1_M', 'IQ1_S', 'IQ2_S', 'IQ2_XS', 'IQ2_XXS', 'IQ3_S', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS', 'MXFP4', 'NVFP4', 'Q1_0', 'Q2_K', 'Q3_K', 'Q4_0', 'Q4_1', 'Q4_K', 'Q5_0', 'Q5_1', 'Q5_K', 'Q6_K', 'Q8_0', 'Q8_1', 'Q8_K', 'TQ1_0', 'TQ2_0'],
        help="Tensor quantization",
    )

    args = parser.parse_args()
    convert(args.input, args.output, args.architecture, args.type)
