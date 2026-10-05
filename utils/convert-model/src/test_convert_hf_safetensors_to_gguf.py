#!/usr/bin/env python3
"""
Synthetic end-to-end test for convert_hf_safetensors_to_gguf.py (no network).

Builds a fake HF-style repo:

    test-synthetic/
    ├── model_index.json
    ├── scheduler/                     (no weights -> must be ignored)
    ├── tokenizer/                     (no weights -> must be ignored)
    ├── transformer/
    │   ├── config.json                (_class_name = FluxTransformer2DModel)
    │   ├── model.safetensors.index.json
    │   ├── model-00001-of-00002.safetensors
    │   ├── model-00002-of-00002.safetensors
    │   └── extra.safetensors          (not referenced by the index)
    ├── vae/
    │   ├── config.json                (_class_name = AutoencoderKLFlux2)
    │   └── diffusion_pytorch_model.safetensors
    └── text_encoder/
        ├── config.json                (architectures = ["Qwen3ForCausalLM"])
        └── model.safetensors

Runs the converter and verifies the GGUF files (arch, names, order, shapes,
dtypes incl. real BF16, exact values) and the removal of the safetensors.

Run with:
    python -m unittest test_convert_hf_safetensors_to_gguf -v
"""
import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import convert_hf_safetensors_to_gguf as conv
from gguf import GGUFReader, GGMLQuantizationType, dequantize

ROOT = HERE / "test-synthetic"
ROOT_KEEP = HERE / "test-synthetic-keep"
ROOT_MISSING = HERE / "test-synthetic-missing"
ROOT_MAP = HERE / "test-synthetic-map"
ROOT_GLOBAL = HERE / "test-synthetic-global"


def field_str(reader, key):
    f = reader.get_field(key)
    return f.contents() if f is not None else None


def tensor_by_name(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    return None


def tensor_values(t):
    if t.tensor_type == GGMLQuantizationType.BF16:
        return dequantize(t.data, GGMLQuantizationType.BF16)
    return t.data.astype(np.float32)


def logical_shape(t):
    return tuple(reversed(t.shape.tolist()))


def make_component_dirs(root: Path):
    """
    Build a fake HF repo layout under `root` and return the original
    tensors so converted values can be compared against them.
    """
    (root / "scheduler").mkdir(parents=True)
    (root / "tokenizer").mkdir()

    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}))
    (root / "tokenizer" / "tokenizer.json").write_text("{}")

    tr = root / "transformer"
    tr.mkdir()
    (tr / "config.json").write_text(
        json.dumps({"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128}))

    s1 = {
        "img_in.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "transformer_blocks.0.ff.net.2.weight": torch.randn(8, 32, dtype=torch.float16),
    }
    s2 = {
        "time_embed.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "transformer_blocks.0.attn.to_q.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "transformer_blocks.0.norm1.weight": torch.randn(8, dtype=torch.float32),
    }
    save_file(s1, tr / "model-00001-of-00002.safetensors")
    save_file(s2, tr / "model-00002-of-00002.safetensors")
    extra = {"extra.buffer": torch.randn(2, 2, dtype=torch.float32)}
    save_file(extra, tr / "extra.safetensors")

    (tr / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {
            "transformer_blocks.0.attn.to_q.weight": "model-00002-of-00002.safetensors",
            "img_in.weight": "model-00001-of-00002.safetensors",
            "time_embed.0.weight": "model-00002-of-00002.safetensors",
            "transformer_blocks.0.ff.net.2.weight": "model-00001-of-00002.safetensors",
            "transformer_blocks.0.norm1.weight": "model-00002-of-00002.safetensors",
        },
    }))

    vae = root / "vae"
    vae.mkdir()
    (vae / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLFlux2"}))
    vae_tensors = {
        "encoder.conv_in.weight": torch.randn(4, 4, 3, 3, dtype=torch.float16),
        "decoder.up.0.residual.0.gamma": torch.randn(4, dtype=torch.float32),
    }
    save_file(vae_tensors, vae / "diffusion_pytorch_model.safetensors")

    te = root / "text_encoder"
    te.mkdir()
    (te / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 16}))
    te_tensors = {
        "model.embed_tokens.weight": torch.randn(128, 16, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(128, 16, dtype=torch.bfloat16),
    }
    save_file(te_tensors, te / "model.safetensors")

    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "Flux2KleinPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
    }))

    return s1, s2, extra, vae_tensors, te_tensors


def run_conversion(root: Path, keep: bool, type_spec=None):
    components = conv.discover_components(root)
    for c in components:
        conv.convert_component(c, type_spec, None, "test-org/test-synthetic", None, keep)
    return components


class ConverterFlowTest(unittest.TestCase):
    """
    Main flow: build the repo, convert with safetensors removal, and verify
    discovery, GGUF contents, metadata and file cleanup.
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        if ROOT.exists():
            shutil.rmtree(ROOT)

        cls.s1, cls.s2, cls.extra, cls.vae_tensors, cls.te_tensors = make_component_dirs(ROOT)
        cls.components = run_conversion(ROOT, keep=False)

    @classmethod
    def tearDownClass(cls):
        if ROOT.exists():
            shutil.rmtree(ROOT)

    def gguf(self, component: str) -> GGUFReader:
        path = ROOT / component / "model.gguf"
        self.assertTrue(path.exists(), f"missing {path}")
        return GGUFReader(str(path))

    def test_component_discovery(self):
        names = [c.name for c in self.components]
        self.assertEqual(names, ["text_encoder", "transformer", "vae"])

    def test_weightless_folders_ignored(self):
        names = {c.name for c in self.components}
        self.assertFalse({"scheduler", "tokenizer"} & names)

    def test_transformer_gguf_arch_from_class_name(self):
        self.assertEqual(field_str(self.gguf("transformer"), "general.architecture"),
                         "FluxTransformer2DModel")

    def test_transformer_tensor_count(self):
        # 2 shards (5 tensors, index.json present) + extra.safetensors (1 tensor)
        self.assertEqual(len(self.gguf("transformer").tensors), 6)

    def test_transformer_tensor_order(self):
        # order follows the index weight_map; unindexed files come last
        order = [t.name for t in self.gguf("transformer").tensors]
        self.assertEqual(order, [
            "transformer_blocks.0.attn.to_q.weight",
            "time_embed.0.weight",
            "transformer_blocks.0.norm1.weight",
            "img_in.weight",
            "transformer_blocks.0.ff.net.2.weight",
            "extra.buffer",
        ])

    def test_bf16_kept_as_gguf_bf16(self):
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.attn.to_q.weight")
        self.assertIsNotNone(t)
        self.assertEqual(t.tensor_type, GGMLQuantizationType.BF16)
        self.assertEqual(logical_shape(t), (16, 8))
        self.assertTrue(np.array_equal(
            tensor_values(t), self.s2["transformer_blocks.0.attn.to_q.weight"].float().numpy()))

    def test_f16_kept_as_gguf_f16(self):
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.ff.net.2.weight")
        self.assertIsNotNone(t)
        self.assertEqual(t.tensor_type, GGMLQuantizationType.F16)
        self.assertTrue(np.array_equal(
            t.data.astype(np.float16), self.s1["transformer_blocks.0.ff.net.2.weight"].numpy()))

    def test_f32_kept_as_gguf_f32(self):
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.norm1.weight")
        self.assertIsNotNone(t)
        self.assertEqual(t.tensor_type, GGMLQuantizationType.F32)
        self.assertTrue(np.array_equal(
            t.data.astype(np.float32), self.s2["transformer_blocks.0.norm1.weight"].numpy()))

    def test_unindexed_safetensors_processed(self):
        self.assertIsNotNone(tensor_by_name(self.gguf("transformer"), "extra.buffer"))

    def test_vae_gguf(self):
        r = self.gguf("vae")
        self.assertEqual(field_str(r, "general.architecture"), "AutoencoderKLFlux2")
        self.assertEqual(len(r.tensors), 2)

        t = tensor_by_name(r, "encoder.conv_in.weight")
        self.assertIsNotNone(t)
        self.assertEqual(t.tensor_type, GGMLQuantizationType.F16)
        self.assertEqual(logical_shape(t), (4, 4, 3, 3))
        self.assertTrue(np.array_equal(
            t.data.astype(np.float16), self.vae_tensors["encoder.conv_in.weight"].numpy()))

    def test_text_encoder_gguf_arch_from_transformers_config(self):
        r = self.gguf("text_encoder")
        self.assertEqual(field_str(r, "general.architecture"), "Qwen3ForCausalLM")
        self.assertEqual(len(r.tensors), 3)

    def test_text_encoder_bf16_roundtrip(self):
        r = self.gguf("text_encoder")
        for name, orig in self.te_tensors.items():
            with self.subTest(tensor=name):
                t = tensor_by_name(r, name)
                self.assertIsNotNone(t)
                self.assertEqual(t.tensor_type, GGMLQuantizationType.BF16)
                self.assertTrue(np.array_equal(tensor_values(t), orig.float().numpy()))

    def test_metadata(self):
        r = self.gguf("transformer")
        self.assertEqual(field_str(r, "general.name"), "test-synthetic")
        self.assertEqual(field_str(r, "general.component"), "transformer")
        self.assertEqual(field_str(r, "general.source.repo_id"), "test-org/test-synthetic")
        self.assertEqual(field_str(r, "general.source.format"), "safetensors")
        self.assertEqual(field_str(r, "general.source.layout"), "diffusers")

    def test_original_safetensors_removed(self):
        for comp_dir in (ROOT / "transformer", ROOT / "vae", ROOT / "text_encoder"):
            with self.subTest(component=comp_dir.name):
                self.assertEqual(list(comp_dir.glob("*.safetensors")), [])
                self.assertEqual(list(comp_dir.glob("*.safetensors.index.json")), [])
                self.assertTrue((comp_dir / "config.json").exists())

    def test_non_weight_files_kept(self):
        self.assertTrue((ROOT / "tokenizer" / "tokenizer.json").exists())
        self.assertTrue((ROOT / "scheduler" / "scheduler_config.json").exists())
        self.assertTrue((ROOT / "model_index.json").exists())


class KeepSafetensorsTest(unittest.TestCase):
    """
    --keep-safetensors mode: GGUFs are created but the originals remain.
    """

    @classmethod
    def setUpClass(cls):
        if ROOT_KEEP.exists():
            shutil.rmtree(ROOT_KEEP)

        make_component_dirs(ROOT_KEEP)
        cls.components = run_conversion(ROOT_KEEP, keep=True)

    @classmethod
    def tearDownClass(cls):
        if ROOT_KEEP.exists():
            shutil.rmtree(ROOT_KEEP)

    def test_gguf_created(self):
        self.assertTrue((ROOT_KEEP / "transformer" / "model.gguf").exists())

    def test_safetensors_preserved(self):
        self.assertEqual(len(list((ROOT_KEEP / "transformer").glob("*.safetensors"))), 3)

    def test_index_preserved(self):
        self.assertTrue((ROOT_KEEP / "transformer" / "model.safetensors.index.json").exists())


class MissingShardTest(unittest.TestCase):
    """
    Index references a shard that does not exist: warn, skip those tensors,
    still convert the rest and clean up the files that were processed.
    """

    @classmethod
    def setUpClass(cls):
        if ROOT_MISSING.exists():
            shutil.rmtree(ROOT_MISSING)

        make_component_dirs(ROOT_MISSING)
        (ROOT_MISSING / "transformer" / "model-00002-of-00002.safetensors").unlink()
        (ROOT_MISSING / "transformer" / "extra.safetensors").unlink()
        shutil.rmtree(ROOT_MISSING / "text_encoder")
        shutil.rmtree(ROOT_MISSING / "vae")

        cls.components = run_conversion(ROOT_MISSING, keep=False)

    @classmethod
    def tearDownClass(cls):
        if ROOT_MISSING.exists():
            shutil.rmtree(ROOT_MISSING)

    def test_missing_shard_tensors_skipped(self):
        path = ROOT_MISSING / "transformer" / "model.gguf"
        self.assertTrue(path.exists(), f"missing {path}")
        names = [t.name for t in GGUFReader(str(path)).tensors]
        self.assertEqual(names, ["img_in.weight", "transformer_blocks.0.ff.net.2.weight"])

    def test_existing_shard_removed(self):
        self.assertEqual(list((ROOT_MISSING / "transformer").glob("model-00001*.safetensors")), [])


class TypeSpecTest(unittest.TestCase):
    """
    --type parsing and per-tensor quantization resolution (no conversion).
    """

    def test_none_keeps_original_dtype(self):
        spec = conv.parse_type_spec(None)
        self.assertIsNone(spec.default)
        self.assertEqual(spec.patterns, {})
        self.assertEqual(conv.resolve_tensor_qtype("a.b", torch.bfloat16, spec),
                         GGMLQuantizationType.BF16)
        self.assertEqual(conv.resolve_tensor_qtype("a.b", torch.float32, spec),
                         GGMLQuantizationType.F32)

    def test_plain_name_applies_to_every_tensor(self):
        spec = conv.parse_type_spec("Q8_0")
        self.assertIs(spec.default, GGMLQuantizationType.Q8_0)
        self.assertEqual(spec.patterns, {})
        self.assertEqual(conv.resolve_tensor_qtype("whatever", torch.bfloat16, spec),
                         GGMLQuantizationType.Q8_0)

    def test_map_exact_beats_pattern(self):
        spec = conv.parse_type_spec(json.dumps({
            "*": "F16",
            "img_in.*": "F32",
            "img_in.weight": "BF16",
        }))
        self.assertEqual(conv.resolve_tensor_qtype("img_in.weight", torch.float16, spec),
                         GGMLQuantizationType.BF16)
        self.assertEqual(conv.resolve_tensor_qtype("img_in.bias", torch.float16, spec),
                         GGMLQuantizationType.F32)
        self.assertEqual(conv.resolve_tensor_qtype("text.weight", torch.float16, spec),
                         GGMLQuantizationType.F16)

    def test_map_most_specific_pattern_wins(self):
        # The broader pattern is listed first: specificity must win over order.
        spec = conv.parse_type_spec(json.dumps({
            "blocks.*": "F16",
            "blocks.*.norm*": "F32",
        }))
        self.assertEqual(conv.resolve_tensor_qtype("blocks.0.norm1.weight", torch.bfloat16, spec),
                         GGMLQuantizationType.F32)
        self.assertEqual(conv.resolve_tensor_qtype("blocks.0.attn.q.weight", torch.bfloat16, spec),
                         GGMLQuantizationType.F16)

    def test_map_unmatched_keeps_dtype(self):
        spec = conv.parse_type_spec(json.dumps({"foo.*": "Q8_0"}))
        self.assertEqual(conv.resolve_tensor_qtype("bar.weight", torch.float16, spec),
                         GGMLQuantizationType.F16)

    def test_at_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"*": "Q8_0"}))
            spec = conv.parse_type_spec(f"@{path}")
            self.assertIs(spec.patterns["*"], GGMLQuantizationType.Q8_0)
            self.assertIsNone(spec.default)

    def test_unknown_type_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            conv.parse_type_spec("Q9_9")
        with self.assertRaises(argparse.ArgumentTypeError):
            conv.parse_type_spec(json.dumps({"*": "Q9_9"}))
        with self.assertRaises(argparse.ArgumentTypeError):
            conv.parse_type_spec("@/nonexistent/map.json")

    def test_invalid_json_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            conv.parse_type_spec('{"*": "Q8_0"')
        with self.assertRaises(argparse.ArgumentTypeError):
            conv.parse_type_spec(json.dumps(["Q8_0"]))


class PerTensorQuantizationTest(unittest.TestCase):
    """
    --type as a JSON object: per-tensor quantization end to end.

    Map:
      "transformer_blocks.0.ff.net.2.weight": "Q8_0"  (exact, beats the pattern)
      "transformer_blocks.*": "F32"
    Everything else keeps its original dtype.
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)

        if ROOT_MAP.exists():
            shutil.rmtree(ROOT_MAP)

        cls.s1, cls.s2, cls.extra, cls.vae_tensors, cls.te_tensors = make_component_dirs(ROOT_MAP)
        spec = conv.parse_type_spec(json.dumps({
            "transformer_blocks.0.ff.net.2.weight": "Q8_0",
            "transformer_blocks.*": "F32",
        }))
        cls.components = run_conversion(ROOT_MAP, keep=False, type_spec=spec)

    @classmethod
    def tearDownClass(cls):
        if ROOT_MAP.exists():
            shutil.rmtree(ROOT_MAP)

    def gguf(self, component: str) -> GGUFReader:
        return GGUFReader(str(ROOT_MAP / component / "model.gguf"))

    def test_transformer_per_tensor_types(self):
        expected = {
            "transformer_blocks.0.attn.to_q.weight": GGMLQuantizationType.F32,
            "time_embed.0.weight": GGMLQuantizationType.BF16,
            "transformer_blocks.0.norm1.weight": GGMLQuantizationType.F32,
            "img_in.weight": GGMLQuantizationType.BF16,
            "transformer_blocks.0.ff.net.2.weight": GGMLQuantizationType.Q8_0,
            "extra.buffer": GGMLQuantizationType.F32,
        }
        for t in self.gguf("transformer").tensors:
            with self.subTest(tensor=t.name):
                self.assertEqual(t.tensor_type, expected[t.name])

    def test_file_type_metadata(self):
        self.assertEqual(field_str(self.gguf("transformer"), "general.file_type"),
                         "BF16, F32, Q8_0")
        self.assertEqual(field_str(self.gguf("vae"), "general.file_type"), "F16, F32")
        self.assertEqual(field_str(self.gguf("text_encoder"), "general.file_type"), "BF16")

    def test_q8_0_values(self):
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.ff.net.2.weight")
        self.assertIsNotNone(t)
        deq = dequantize(t.data, GGMLQuantizationType.Q8_0)
        orig = self.s1["transformer_blocks.0.ff.net.2.weight"].float().numpy()
        self.assertEqual(deq.shape, orig.shape)
        # Q8_0 block quantization error is bounded per block: |err| <= amax/254
        self.assertLess(np.abs(deq - orig).max(), 1e-2 * np.abs(orig).max())

    def test_f32_pattern_values(self):
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.attn.to_q.weight")
        self.assertIsNotNone(t)
        self.assertTrue(np.array_equal(
            t.data.astype(np.float32),
            self.s2["transformer_blocks.0.attn.to_q.weight"].float().numpy()))

    def test_unmatched_keeps_bf16(self):
        t = tensor_by_name(self.gguf("transformer"), "img_in.weight")
        self.assertIsNotNone(t)
        self.assertEqual(t.tensor_type, GGMLQuantizationType.BF16)
        self.assertTrue(np.array_equal(
            tensor_values(t), self.s1["img_in.weight"].float().numpy()))


class GlobalTypeTest(unittest.TestCase):
    """
    --type as a single type name: every tensor gets that type.
    """

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2)

        if ROOT_GLOBAL.exists():
            shutil.rmtree(ROOT_GLOBAL)

        cls.s1, cls.s2, cls.extra, cls.vae_tensors, cls.te_tensors = make_component_dirs(ROOT_GLOBAL)
        cls.components = run_conversion(ROOT_GLOBAL, keep=False, type_spec="F16")

    @classmethod
    def tearDownClass(cls):
        if ROOT_GLOBAL.exists():
            shutil.rmtree(ROOT_GLOBAL)

    def gguf(self, component: str) -> GGUFReader:
        return GGUFReader(str(ROOT_GLOBAL / component / "model.gguf"))

    def test_all_tensors_f16(self):
        for component in ("transformer", "vae", "text_encoder"):
            for t in self.gguf(component).tensors:
                with self.subTest(component=component, tensor=t.name):
                    self.assertEqual(t.tensor_type, GGMLQuantizationType.F16)

    def test_file_type_metadata(self):
        for component in ("transformer", "vae", "text_encoder"):
            with self.subTest(component=component):
                self.assertEqual(field_str(self.gguf(component), "general.file_type"), "F16")

    def test_values(self):
        # bf16 -> f16 is lossless for these magnitudes
        for name, orig in self.te_tensors.items():
            t = tensor_by_name(self.gguf("text_encoder"), name)
            self.assertIsNotNone(t)
            self.assertTrue(np.array_equal(
                t.data.astype(np.float16), orig.half().numpy()), name)

        # f32 -> f16 is only approximate
        t = tensor_by_name(self.gguf("transformer"), "transformer_blocks.0.norm1.weight")
        self.assertIsNotNone(t)
        self.assertTrue(np.allclose(
            t.data.astype(np.float32),
            self.s2["transformer_blocks.0.norm1.weight"].numpy(),
            rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
