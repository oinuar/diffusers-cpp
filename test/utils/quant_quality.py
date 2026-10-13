"""
Measure the end-to-end quality degradation of Flux2Klein quantization specs.

Builds the same tiny pipeline as test/src/diffusers/pipelines/flux2/Flux2KleinPipeline.py
test_call, runs it with the original float32 weights (the reference), and again
with every candidate spec: each parameter is quantized and dequantized the way
utils/convert-model does, and the resulting float32 values are loaded back into
the Python reference.

For every spec the script reports, against the float32 reference:
  * the text encoder prompt embeddings (relative L2 error + max abs error),
    computed with identical token inputs;
  * the transformer's first denoising-step noise prediction (relative L2 error
    + max abs error), computed with identical latents and the float32 prompt
    embeddings (so the transformer error is isolated from the encoder error);
  * the final image (max/mean absolute pixel difference, 0-255 scale).

Run from the test/ directory:

    uv run python utils/quant_quality.py
"""

import json
import os
import sys
import time

# Allow running the script directly: uv run python utils/quant_quality.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from diffusers import (
    AutoencoderKLFlux2,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2TokenizerFast

from utils import QuantSpec, quantize_to_bytes, dequantize_to_float32
from gguf import GGMLQuantizationType, quant_shape_to_byte_shape

PLAIN_DTYPE_BYTES = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F64: 8,
    GGMLQuantizationType.I8: 1,
    GGMLQuantizationType.I16: 2,
    GGMLQuantizationType.I32: 4,
    GGMLQuantizationType.I64: 8,
    GGMLQuantizationType.F16: 2,
}


def build_env(tokenizer_dir):
    # Same configuration as test_call: a model in the shape class of the real
    # FLUX.2-klein-9B (4D RoPE, GQA text encoder, double+single stream blocks,
    # multi-block VAE with mid-block attention).
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=32,
        out_channels=32,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=96,
        timestep_guidance_channels=32,
        axes_dims_rope=(8, 8, 8, 8),
        guidance_embeds=False,
    )

    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        latent_channels=8,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(8, 16, 32),
        layers_per_block=2,
        norm_num_groups=8,
    )

    text_config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
    )

    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_dir)
    text_encoder = Qwen3ForCausalLM(text_config)
    scheduler = FlowMatchEulerDiscreteScheduler()

    pipe = Flux2KleinPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    generator = torch.Generator()

    packed_h, packed_w = 4, 4
    height = packed_h * 2 * pipe.vae_scale_factor
    width = packed_w * 2 * pipe.vae_scale_factor

    init_latents = torch.randn(
        1,
        pipe.vae.config.latent_channels * 4,
        packed_h,
        packed_w,
        generator=generator,
        dtype=torch.float32,
    )

    run_kwargs = dict(
        prompt="hello world",
        height=height,
        width=width,
        num_inference_steps=4,
        guidance_scale=1.0,
        max_sequence_length=128,
    )

    return pipe, init_latents, run_kwargs


def run_pipeline(pipe, init_latents, run_kwargs):
    output = pipe(latents=init_latents, **run_kwargs)
    return [torch.from_numpy(np.array(image)).float() for image in output.images]


def tensor_bytes(tensor, qtype):
    shape = tuple(tensor.shape)
    if qtype in PLAIN_DTYPE_BYTES:
        n = 1
        for d in shape:
            n *= d
        return n * PLAIN_DTYPE_BYTES[qtype]

    n = 1
    for d in quant_shape_to_byte_shape(shape, qtype):
        n *= d
    return n


def storage_bytes(pipe, spec_value):
    """Bytes the GGUF would store for every parameter under the spec."""
    spec = QuantSpec.parse(spec_value)

    total = 0
    per_component = {}

    for model, prefix in (
        (pipe.transformer, "transformer"),
        (pipe.vae, "vae"),
        (pipe.text_encoder, "text_encoder"),
    ):
        component = 0
        for name, tensor in model.named_parameters():
            qtype = spec.resolve(name, prefix) or GGMLQuantizationType.F32
            component += tensor_bytes(tensor, qtype)
        total += component
        per_component[prefix] = component

    return total, per_component


def quantize_model(model, spec, prefix):
    """
    Mirror of TestCase.quant_params, but without writing parameter files.
    Returns the dequantized state of every quantized parameter.
    """
    state = {}

    for name, tensor, is_buffer in (
        [(n, t, False) for n, t in model.named_parameters()]
        + [(n, t, True) for n, t in model.named_buffers()]
    ):
        qtype = None if is_buffer else spec.resolve(name, prefix)

        if qtype is None or qtype == GGMLQuantizationType.F32:
            continue

        raw = quantize_to_bytes(tensor, qtype)
        state[name] = dequantize_to_float32(raw, qtype, tuple(tensor.shape))

    return state


def apply_state(model, state):
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


def rel_l2(a, b):
    return ((a - b).norm() / a.norm()).item()


class TransformerSpy:
    """
    Records the first transformer forward call of a pipeline run so the
    noise prediction can be compared under identical inputs.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.orig = pipe.transformer.forward
        self.first = None

    def start(self):
        def spy(*args, **kwargs):
            out = self.orig(*args, **kwargs)
            if self.first is None:
                first = out[0] if isinstance(out, (tuple, list)) else out
                self.first = (first, args, kwargs)
            return out

        self.pipe.transformer.forward = spy

    def stop(self):
        self.pipe.transformer.forward = self.orig

    def replay(self):
        """Run the first recorded call on the current weights and return the output."""
        out, args, kwargs = self.first
        out = self.orig(*args, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out


def main():
    tokenizer_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "src", "transformers", "models", "qwen2"
    ))

    pipe, init_latents, run_kwargs = build_env(tokenizer_dir)

    models = {
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
    }
    # Clone: state_dict() shares storage with the live parameters.
    original_state = {
        name: {k: v.detach().clone() for k, v in model.state_dict().items()}
        for name, model in models.items()
    }

    def restore():
        for name, model in models.items():
            model.load_state_dict(original_state[name])

    def prompt_embeds():
        embeds, _ = pipe.encode_prompt(
            prompt=run_kwargs["prompt"],
            num_images_per_prompt=1,
            max_sequence_length=run_kwargs["max_sequence_length"],
        )
        return embeds.float()

    spy = TransformerSpy(pipe)

    # ------------------------------------------------------------
    # Reference (float32) signals.
    # ------------------------------------------------------------
    restore()
    ref_embeds = prompt_embeds()

    spy.start()
    ref_images = run_pipeline(pipe, init_latents, run_kwargs)
    ref_pred = spy.first[0]
    spy.stop()

    f32_total, _ = storage_bytes(pipe, None)

    # ------------------------------------------------------------
    # Candidate specs.
    # ------------------------------------------------------------
    attn_norms = {
        "transformer_blocks.*.attn.norm_q.weight": "F32",
        "transformer_blocks.*.attn.norm_k.weight": "F32",
        "single_transformer_blocks.*.attn.norm_k.weight": "F32",
    }
    te_norms = {
        "model.layers.*.input_layernorm.weight": "F32",
        "model.layers.*.post_attention_layernorm.weight": "F32",
        "model.layers.*.self_attn.q_norm.weight": "F32",
        "model.layers.*.self_attn.k_norm.weight": "F32",
        "model.norm.weight": "F32",
    }

    def spec_obj(transformer_extra, text_extra=None):
        return {
            "transformer": {"*": "Q8_0", **attn_norms, **transformer_extra},
            "text_encoder": {"*": "Q8_0", **te_norms, **(text_extra or {})},
            "vae": {"*": "F32"},
        }

    specs = {
        # the checked-in config (Q8_0 everywhere except attention norms)
        "current": spec_obj({}),
        # modulation (shift/scale/gate) of the transformer blocks
        "t-mod": spec_obj({
            "double_stream_modulation_img.linear.weight": "F32",
            "double_stream_modulation_txt.linear.weight": "F32",
            "single_stream_modulation.linear.weight": "F32",
        }),
        # input projections + timestep embeddings
        "t-in": spec_obj({
            "x_embedder.weight": "F32",
            "context_embedder.weight": "F32",
            "time_guidance_embed.timestep_embedder.linear_1.weight": "F32",
            "time_guidance_embed.timestep_embedder.linear_2.weight": "F32",
        }),
        # final output projection
        "t-out": spec_obj({
            "norm_out.linear.weight": "F32",
            "proj_out.weight": "F32",
        }),
        # all double-block attention projections
        "t-attn": spec_obj({
            "transformer_blocks.*.attn.to_q.weight": "F32",
            "transformer_blocks.*.attn.to_k.weight": "F32",
            "transformer_blocks.*.attn.to_v.weight": "F32",
            "transformer_blocks.*.attn.to_out.0.weight": "F32",
            "transformer_blocks.*.attn.add_q_proj.weight": "F32",
            "transformer_blocks.*.attn.add_k_proj.weight": "F32",
            "transformer_blocks.*.attn.add_v_proj.weight": "F32",
            "transformer_blocks.*.attn.to_add_out.weight": "F32",
        }),
        # all double-block feedforwards
        "t-ff": spec_obj({
            "transformer_blocks.*.ff.linear_in.weight": "F32",
            "transformer_blocks.*.ff.linear_out.weight": "F32",
            "transformer_blocks.*.ff_context.linear_in.weight": "F32",
            "transformer_blocks.*.ff_context.linear_out.weight": "F32",
        }),
        # all single-block tensors
        "t-single": spec_obj({
            "single_transformer_blocks.*.attn.to_qkv_mlp_proj.q.weight": "F32",
            "single_transformer_blocks.*.attn.to_qkv_mlp_proj.k.weight": "F32",
            "single_transformer_blocks.*.attn.to_qkv_mlp_proj.v.weight": "F32",
            "single_transformer_blocks.*.attn.to_qkv_mlp_proj.mlp_in.weight": "F32",
            "single_transformer_blocks.*.attn.to_out.attn.weight": "F32",
            "single_transformer_blocks.*.attn.to_out.mlp.weight": "F32",
        }),
        # text encoder embedding table
        "te-embed": spec_obj({}, {"model.embed_tokens.weight": "F32"}),
        # text encoder attention projections
        "te-attn": spec_obj({}, {
            "model.layers.*.self_attn.q_proj.weight": "F32",
            "model.layers.*.self_attn.k_proj.weight": "F32",
            "model.layers.*.self_attn.v_proj.weight": "F32",
            "model.layers.*.self_attn.o_proj.weight": "F32",
        }),
        # text encoder MLPs
        "te-mlp": spec_obj({}, {
            "model.layers.*.mlp.gate_proj.weight": "F32",
            "model.layers.*.mlp.up_proj.weight": "F32",
            "model.layers.*.mlp.down_proj.weight": "F32",
        }),
    }

    # worst case: everything Q8_0, not even the attention norms
    specs["worst"] = {
        "transformer": {"*": "Q8_0"},
        "text_encoder": {"*": "Q8_0"},
        "vae": {"*": "F32"},
    }

    # Composite candidates for the final config.
    t_safe = {
        "double_stream_modulation_img.linear.weight": "F32",
        "double_stream_modulation_txt.linear.weight": "F32",
        "single_stream_modulation.linear.weight": "F32",
        "x_embedder.weight": "F32",
        "context_embedder.weight": "F32",
        "time_guidance_embed.timestep_embedder.linear_1.weight": "F32",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "F32",
        "norm_out.linear.weight": "F32",
        "proj_out.weight": "F32",
    }
    specs["c1-te-f32"] = {
        "transformer": {"*": "Q8_0", **attn_norms, **t_safe},
        "text_encoder": {"*": "F32"},
        "vae": {"*": "F32"},
    }
    te_attn_f32 = {
        "model.layers.*.self_attn.q_proj.weight": "F32",
        "model.layers.*.self_attn.k_proj.weight": "F32",
        "model.layers.*.self_attn.v_proj.weight": "F32",
        "model.layers.*.self_attn.o_proj.weight": "F32",
    }
    specs["c2-te-attn-f32"] = {
        "transformer": {"*": "Q8_0", **attn_norms, **t_safe},
        "text_encoder": {"*": "Q8_0", **te_norms, **te_attn_f32},
        "vae": {"*": "F32"},
    }
    specs["c2b-te-attn-embed-f32"] = {
        "transformer": {"*": "Q8_0", **attn_norms, **t_safe},
        "text_encoder": {"*": "Q8_0", **te_norms, **te_attn_f32, "model.embed_tokens.weight": "F32"},
        "vae": {"*": "F32"},
    }
    specs["c3-t-safe-only"] = {
        "transformer": {"*": "Q8_0", **attn_norms, **t_safe},
        "text_encoder": {"*": "Q8_0", **te_norms},
        "vae": {"*": "F32"},
    }

    print(
        f"{'spec':<10} {'embed rel':>10} {'embed max':>10} {'pred rel':>10} "
        f"{'pred max':>10} {'img max':>8} {'img mean':>9} {'MiB':>8} {'%F32':>6}"
    )

    for label, spec in specs.items():
        spec_value = json.dumps(spec)
        spec = QuantSpec.parse(spec_value)

        restore()

        for name, model in models.items():
            apply_state(model, quantize_model(model, spec, name))

        # Text encoder error, identical tokens.
        embeds = prompt_embeds()
        e_rel = rel_l2(ref_embeds, embeds)
        e_max = (ref_embeds - embeds).abs().max().item()

        # Transformer error, identical inputs (float32 embeddings).
        pred = spy.replay()
        p_rel = rel_l2(ref_pred, pred)
        p_max = (ref_pred - pred).abs().max().item()

        # End-to-end image error.
        images = run_pipeline(pipe, init_latents, run_kwargs)
        diff = (torch.stack(images) - torch.stack(ref_images)).abs()
        i_max = diff.max().item()
        i_mean = diff.mean().item()

        total, _ = storage_bytes(pipe, spec_value)

        print(
            f"{label:<10} {e_rel:10.2e} {e_max:10.3e} {p_rel:10.2e} "
            f"{p_max:10.3e} {i_max:8.3f} {i_mean:9.4f} {total / 2**20:8.1f} "
            f"{100 * total / f32_total:6.1f}",
            flush=True,
        )

    print(f"\nF32 storage: {f32_total / 2**20:.1f} MiB; image shape: {tuple(ref_images[0].shape)}")


if __name__ == "__main__":
    main()
