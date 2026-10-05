#!/usr/bin/env python3
"""
Convert a Hugging Face model repository into GGUF files.

huggingface_hub downloads the model snapshot into the directory that
contains this script, then every folder in the snapshot that contains
.safetensors files (e.g. transformer/, vae/, text_encoder/) is treated as
one model component and converted into its own `model.gguf` written into
the same folder. After a component is converted, its original .safetensors
files (and their index file) are removed, so the result looks like:

    HF repo
    ├── transformer/
    │   └── model.gguf
    ├── vae/
    │   └── model.gguf
    └── text_encoder/
        ├── model.gguf
        └── config.json

The component's HF config.json is loaded to determine the GGUF
architecture. By default tensors keep their original dtype (bfloat16
weights are stored as real GGUF BF16); --type can override the
quantization, either for every tensor or individually per tensor via a
JSON object of name patterns; the JSON may also hold one mapping per
model component (see --type help).

Usage:
    python convert_hf_safetensors_to_gguf.py <huggingface-model-id> [options]

Examples:
    python convert_hf_safetensors_to_gguf.py black-forest-labs/FLUX.1-klein-base-9b

    # every tensor to Q8_0
    python convert_hf_safetensors_to_gguf.py org/model --type Q8_0

    # per tensor: Q8_0 by default, embeddings in BF16
    python convert_hf_safetensors_to_gguf.py org/model \
        --type '{"*": "Q8_0", "model.embed_tokens.weight": "BF16"}'

    # per component: different mapping per model component
    python convert_hf_safetensors_to_gguf.py org/model \
        --type '{"transformer": {"*": "Q8_0"}, "text_encoder": "BF16"}'
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType, quantize


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "models"


def map_tensor_name(name):
    """
    Translate Diffusers tensor names into the tensor names expected by GGUF.

    Returning None skips a tensor.
    """
    return name


DTYPE_TO_QTYPE = {
    torch.bfloat16: GGMLQuantizationType.BF16,
    torch.float64: GGMLQuantizationType.F64,
    torch.float32: GGMLQuantizationType.F32,
    torch.float16: GGMLQuantizationType.F16,
    torch.int8: GGMLQuantizationType.I8,
    torch.int16: GGMLQuantizationType.I16,
    torch.int32: GGMLQuantizationType.I32,
    torch.int64: GGMLQuantizationType.I64,
}


@dataclass
class TypeSpec:
    """
    Parsed --type value.

    default  -- a single quantization type applied to every tensor
    patterns -- per-tensor map: tensor name pattern -> quantization type
    """
    default: Optional[GGMLQuantizationType] = None
    patterns: dict = field(default_factory=dict)


@dataclass
class ComponentSpec:
    """
    Parsed per-component --type value.

    specs    -- component name -> TypeSpec (of that component)
    fallback -- TypeSpec from the "*" entry, applied to components without
                an explicit entry (None if absent)
    """
    specs: dict = field(default_factory=dict)
    fallback: Optional[TypeSpec] = None


def quantization_type(name: str) -> GGMLQuantizationType:
    """
    Convert a type name (e.g. "Q8_0") to its GGMLQuantizationType.
    """
    try:
        return GGMLQuantizationType[name]
    except KeyError:
        valid = ", ".join(t.name for t in GGMLQuantizationType)
        raise argparse.ArgumentTypeError(
            f"Unknown quantization type {name!r}. Valid types: {valid}") from None


def parse_type_spec(value) -> "TypeSpec | ComponentSpec":
    """
    Parse a --type value.

    Accepts:
      * None                        -> keep each tensor's original dtype
      * "Q8_0"                      -> one type for every tensor
      * '{"*": "Q8_0", "foo": "BF16"}' -> per-tensor JSON object, applied
        to every component
      * '{"transformer": {"*": "Q8_0"}, "text_encoder": "BF16"}'
                                      -> per-component JSON object: one
        entry per component, each a type name (applied to every tensor of
        that component) or a per-tensor object; the "*" entry is the
        default for components without an explicit entry
      * "@map.json"                 -> per-tensor or per-component JSON
        object read from a file

    A JSON object with only string values is a per-tensor map for all
    components; the per-component form is detected as soon as any value
    is an object.

    Per-tensor objects: keys are tensor names as found in the source
    .safetensors files. Exact names take precedence over patterns;
    patterns use fnmatch syntax (*, ?, [seq]) and, when several match,
    the most specific one (the longest pattern) wins, so a broad "*"
    default can be combined with narrower overrides in any order.
    Tensors matching nothing fall back to the tensor's original dtype.
    """
    if value is None:
        return TypeSpec()

    if isinstance(value, (TypeSpec, ComponentSpec)):
        return value

    obj = None

    if isinstance(value, str):
        text = value.strip()

        if text.startswith("@"):
            path = Path(text[1:])
            try:
                raw = path.read_text()
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    f"Cannot read --type file {path}: {e}") from None
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--type file {path} is not valid JSON: {e}") from None
        elif text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"--type is not valid JSON: {e}") from None
        else:
            # a bare quantization type name applied to every tensor
            return TypeSpec(default=quantization_type(text))
    elif isinstance(value, dict):
        obj = value

    if obj is None:
        raise argparse.ArgumentTypeError(
            "--type must be a quantization type name, a JSON object "
            "of tensor name patterns to type names, or @file.json")

    if isinstance(obj, str):
        return TypeSpec(default=quantization_type(obj))

    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError(
            "--type JSON must be an object mapping tensor name patterns to type names")

    if any(isinstance(item, dict) for item in obj.values()):
        return parse_component_spec(obj)

    return parse_type_map(obj)


def parse_type_map(obj: dict) -> TypeSpec:
    """
    Parse a per-tensor --type JSON object: tensor name pattern -> type name.
    """
    spec = TypeSpec()
    for pattern, type_name in obj.items():
        if not isinstance(pattern, str):
            raise argparse.ArgumentTypeError(
                f"--type pattern {pattern!r} must be a string")
        if not isinstance(type_name, str):
            raise argparse.ArgumentTypeError(
                f"--type value for pattern {pattern!r} must be a quantization type name")
        spec.patterns[pattern] = quantization_type(type_name)

    return spec


def parse_component_spec(obj: dict) -> ComponentSpec:
    """
    Parse a per-component --type JSON object: component name -> spec.

    Each entry is either a quantization type name applied to every tensor
    of that component, or a per-tensor object (patterns -> type names).
    The "*" entry is the default for components without an explicit entry.
    """
    spec = ComponentSpec()
    for name, value in obj.items():
        if isinstance(value, TypeSpec):
            component_spec = value
        elif isinstance(value, dict):
            component_spec = parse_type_map(value)
        elif isinstance(value, str):
            component_spec = TypeSpec(default=quantization_type(value))
        else:
            raise argparse.ArgumentTypeError(
                f"--type component {name!r} must be a quantization type name "
                "or an object of tensor name patterns to type names")
        if name == "*":
            if spec.fallback is not None:
                raise argparse.ArgumentTypeError(
                    "--type has multiple '*' component entries")
            spec.fallback = component_spec
        else:
            spec.specs[name] = component_spec

    return spec


def resolve_component_spec(value, component: "Component") -> TypeSpec:
    """
    Pick the TypeSpec of one component from a parsed --type value.

    value may be None, a global TypeSpec (applies to every component) or a
    ComponentSpec. Unlisted components fall back to the "*" entry, if any,
    and otherwise to an empty spec (original dtypes). The snapshot root
    component additionally matches the key ".".
    """
    if value is None or isinstance(value, TypeSpec):
        return value or TypeSpec()

    for key in (component.name, * (["."] if component.rel == "" else [])):
        if key in value.specs:
            return value.specs[key]

    if value.fallback is not None:
        return value.fallback

    return TypeSpec()


def resolve_tensor_qtype(name, dtype, spec: TypeSpec) -> GGMLQuantizationType:
    """
    Decide the GGUF quantization type of a tensor.

    Priority:
      1. an exact tensor-name entry of the --type JSON object
      2. the most specific --type pattern that matches the name
         (longest pattern wins; ties in the order they appear)
      3. the global --type name
      4. the tensor's original dtype
    """
    qtype = spec.patterns.get(name)

    if qtype is None:
        patterns = sorted(spec.patterns, key=len, reverse=True)
        for pattern in patterns:
            if fnmatch.fnmatchcase(name, pattern):
                qtype = spec.patterns[pattern]
                break

    if qtype is not None:
        return qtype

    if spec.default is not None:
        return spec.default

    try:
        return DTYPE_TO_QTYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype {dtype} for tensor '{name}'") from None


def normalize_tensor(tensor):
    """
    Normalize a torch tensor for GGUF writing.
    """
    tensor = tensor.detach().cpu().contiguous()

    # numpy cannot represent BF16; promote to float32 (lossless) so the
    # value can later be written as F32 or packed into real GGUF BF16.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    return tensor.numpy()


def quantize_tensor(name, tensor, qtype):
    """
    Convert a numpy tensor to the raw bytes a GGUF file stores for `qtype`.
    """
    # Plain dtypes are written as-is.
    if qtype in (
        GGMLQuantizationType.F64,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    ):
        return tensor

    # gguf handles F32/F16 casts, real BF16 packing (returns a uint8 byte
    # blob of shape (*shape, 2)) and all block-quantized types.
    return quantize(tensor, qtype)


@dataclass
class Component:
    name: str                    # folder name relative to the snapshot root
    dir: Path                    # directory holding the .safetensors files
    config: dict                 # parsed config.json (empty dict if absent)
    safetensors: list            # every .safetensors file in the folder
    index_files: list            # every *.safetensors.index.json file
    index: Optional[dict] = None  # parsed index file (weight_map), if any
    rel: str = ""                # path relative to the snapshot root ("" = root)


def load_hf_config(dir: Path) -> dict:
    """
    Load the HF config.json of a component folder, if present.
    """
    config_file = dir / "config.json"

    if not config_file.exists():
        return {}

    with open(config_file, "r") as f:
        return json.load(f)


def discover_components(snapshot_dir: Path) -> list:
    """
    Discover every model/component folder under the snapshot.

    A component is any folder (including the snapshot root itself) that
    directly contains at least one .safetensors file. Folders without
    weights (tokenizer/, scheduler/, ...) are ignored.
    """
    components = []

    dirs = [snapshot_dir]
    dirs += [p for p in snapshot_dir.rglob("*") if p.is_dir() and not p.name.startswith(".")]

    for dir in sorted(set(dirs), key=lambda p: p.relative_to(snapshot_dir).as_posix()):
        safetensors = sorted(dir.glob("*.safetensors"))

        if not safetensors:
            continue

        index_files = sorted(dir.glob("*.safetensors.index.json"))

        index = None
        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)

        rel = dir.relative_to(snapshot_dir).as_posix()
        name = rel if rel else snapshot_dir.name

        components.append(Component(
            name=name,
            dir=dir,
            config=load_hf_config(dir),
            safetensors=safetensors,
            index_files=index_files,
            index=index,
            rel=rel,
        ))

    return components


def architecture_from_config(config: dict) -> Optional[str]:
    """
    Determine the model architecture from an HF config.json.

    Diffusers configs carry `_class_name` (e.g. FluxTransformer2DModel),
    transformers configs carry `architectures` (e.g. ["Qwen3ForCausalLM"]).
    """
    class_name = config.get("_class_name")
    if class_name:
        return str(class_name)

    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])

    return None


def add_component_metadata(writer, file_type, component, model_id, revision):
    """
    Add generic metadata to the GGUF file.
    """
    writer.add_string("general.name", model_id.split("/")[-1] if model_id else component.name)
    writer.add_string("general.component", component.name)
    writer.add_string("general.source.format", "safetensors")
    writer.add_string("general.source.layout", "diffusers")

    if model_id:
        writer.add_string("general.source.repo_id", model_id)

    if revision:
        writer.add_string("general.source.revision", revision)

    if file_type:
        writer.add_string("general.file_type", file_type)


def load_safetensors_model(path, tensor_names=None):
    print(f"LOAD {path}")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        tensor_names = tensor_names if tensor_names else f.keys()

        for name in tensor_names:
            tensor = f.get_tensor(name)
            yield (name, tensor)


def iter_component_tensors(component: Component):
    """
    Yield (name, tensor) for every .safetensors file in the component,
    including sharded checkpoints.

    When a shard index file is present, the tensor order follows its
    weight_map. Any .safetensors files not referenced by the index are
    processed afterwards, in file name order.
    """
    pending = {p.name: p for p in component.safetensors}

    if component.index is not None:
        weight_map = component.index.get("weight_map", {})

        # shard file -> tensor names (in index order)
        shard_map = {}
        for tensor_name, shard_file in weight_map.items():
            shard_map.setdefault(shard_file, []).append(tensor_name)

        for shard_file, tensor_names in shard_map.items():
            shard = pending.pop(shard_file, None)

            if shard is None:
                print(
                    f"WARNING: shard '{shard_file}' referenced by "
                    f"{component.index_files[0].name} is missing, skipping {len(tensor_names)} tensor(s)"
                )
                continue

            yield from load_safetensors_model(shard, tensor_names)

        pending = dict(sorted(pending.items()))

    for shard in pending.values():
        yield from load_safetensors_model(shard)


def file_type_name(used_types) -> Optional[str]:
    """
    Summarize the quantization types actually written into a GGUF file
    for its general.file_type metadata (e.g. "Q8_0" or "BF16, Q8_0").
    """
    if not used_types:
        return None

    return ", ".join(sorted(t.name for t in used_types))


def convert_component(component: Component, type_spec, architecture: Optional[str], model_id: Optional[str], revision: Optional[str], keep_safetensors: bool) -> Path:
    arch = architecture or architecture_from_config(component.config) or component.name
    spec = resolve_component_spec(type_spec, component)

    if isinstance(type_spec, ComponentSpec) and not (
        component.name in type_spec.specs
        or (component.rel == "" and "." in type_spec.specs)
        or type_spec.fallback is not None):
        print(f"NOTE: no --type entry for component {component.name}, keeping original dtypes")

    output_path = component.dir / "model.gguf"

    print(f"\nCONVERT COMPONENT {component.name} "
          f"({len(component.safetensors)} safetensors file(s)) -> {output_path} [arch={arch}]")

    writer = GGUFWriter(path=str(output_path), arch=arch)

    converted = 0
    skipped = 0
    used_types = set()
    matched_patterns = set()

    for name, tensor in iter_component_tensors(component):
        target_name = map_tensor_name(name)

        if target_name is None:
            skipped += 1
            continue

        source_type = tensor.dtype
        dest_type = resolve_tensor_qtype(name, source_type, spec)
        if name in spec.patterns:
            matched_patterns.add(name)
        else:
            for pattern in sorted(spec.patterns, key=len, reverse=True):
                if fnmatch.fnmatchcase(name, pattern):
                    matched_patterns.add(pattern)
                    break
        shape = tensor.shape

        tensor = normalize_tensor(tensor)
        tensor = quantize_tensor(target_name, tensor, dest_type)

        writer.add_tensor(
            name=target_name,
            tensor=tensor,
            raw_dtype=dest_type,
        )

        used_types.add(dest_type)

        converted += 1

        print(f"  CONVERT {name} ({source_type}) -> {dest_type.name} {list(shape)}")

    if spec.patterns and not matched_patterns:
        shown = ", ".join(sorted(spec.patterns)[:5])
        more = ", ..." if len(spec.patterns) > 5 else ""
        print(f"WARNING: no tensor of component {component.name} matched any --type pattern ({shown}{more})")

    add_component_metadata(writer, file_type_name(used_types), component, model_id, revision)

    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if not keep_safetensors:
        for path in component.safetensors:
            path.unlink()
            print(f"REMOVED {path}")

        for path in component.index_files:
            path.unlink()
            print(f"REMOVED {path}")

    print(f"Done: {output_path}")
    print(f"Converted tensors: {converted}")
    print(f"Skipped tensors:   {skipped}")

    return output_path


def download_model(model_id, revision=None, download_dir: Path = SCRIPT_DIR) -> Path:
    """
    Download a Hugging Face model into a directory next to this script.
    """
    download_dir = Path(download_dir) / model_id
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} to {download_dir}...")

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=download_dir,
        allow_patterns=["scheduler/**", "text_encoder/**", "tokenizer/**", "transformer/**", "vae/**"],
    )

    return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model and convert every safetensors component into its own GGUF file."
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model repo id, e.g. black-forest-labs/FLUX.1-klein-base-9b",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision (branch, tag or commit id). Defaults to the repo's main branch.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory the model snapshot is downloaded into (default: the directory containing this script).",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="Force this GGUF architecture for every component. By default it is derived from each component's config.json.",
    )
    parser.add_argument(
        "--type",
        type=parse_type_spec,
        default=None,
        help="Tensor quantization. Either a single type name applied to every tensor "
             "(e.g. Q8_0), or a JSON object. The JSON is either a per-tensor map "
             "of tensor name patterns (from the source .safetensors files) to type "
             'names applied to all components, e.g. \'{"*": "Q8_0", "*.bias": "F16"}\', '
             "or a per-component map of component names to type names or per-tensor "
             "objects (detected when any value is an object), e.g. "
             '\'{"transformer": {"*": "Q8_0"}, "text_encoder": "BF16"}\'; the "*" '
             'component is the default for unlisted components, and the snapshot '
             'root component also matches the key ".". Exact names take precedence '
             "over patterns (fnmatch syntax; when several patterns match, the most "
             "specific one wins), unmatched tensors keep their original dtype, and "
             "'@file.json' reads the map from a file. Defaults to keeping each "
             "tensor's original dtype (bfloat16 stays real GGUF BF16).",
    )
    parser.add_argument(
        "--keep-safetensors",
        action="store_true",
        help="Do not delete the original .safetensors files after conversion.",
    )

    args = parser.parse_args()

    snapshot_dir = download_model(args.model_id, args.revision, args.download_dir)

    components = discover_components(snapshot_dir)

    if not components:
        raise SystemExit(f"ERROR: no .safetensors files found under {snapshot_dir}")

    print(f"\nFound {len(components)} component(s) under {snapshot_dir}:")
    for component in components:
        arch = args.architecture or architecture_from_config(component.config) or component.name
        print(f"  - {component.name} ({len(component.safetensors)} safetensors file(s), arch={arch})")

    for component in components:
        convert_component(
            component,
            args.type,
            args.architecture,
            args.model_id,
            args.revision,
            args.keep_safetensors,
        )

    print(f"\nAll {len(components)} component(s) converted under {snapshot_dir}")


if __name__ == "__main__":
    main()
