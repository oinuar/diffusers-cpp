import json
import os
import tempfile
import pathlib
import torch
from diffusers import (
    AutoencoderKLFlux2,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2TokenizerFast
from utils import TestCase

class TestPipelinesFlux2KleinPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        tokenizer_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "utils" / "convert-model" / "tokenizer"

        if not tokenizer_dir.is_dir():
            raise RuntimeError("No such directory: {tokenizer_dir}. It is required for tokenizer-related tests.")

        cls.tokenizer_dir = str(tokenizer_dir)
        cls.tokenizer_file = str(tokenizer_dir / "tokenizer.json")
        cls.tmpdir = tempfile.TemporaryDirectory()

    def test_embeddings(self):
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,
            joint_attention_dim=24,
            timestep_guidance_channels=8,
            axes_dims_rope=(2, 2),
            guidance_embeds=False,
        )

        vae = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        text_config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )

        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

        text_encoder = Qwen3ForCausalLM(text_config)

        scheduler = FlowMatchEulerDiscreteScheduler()

        pipe = Flux2KleinPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        batch = 1
        prompt = "hello world"
        max_sequence_length = 16

        packed_h = 2
        packed_w = 3

        prompt_embeds, txt_ids = pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=batch,
            max_sequence_length=max_sequence_length,
        )

        latents = torch.zeros(
            batch,
            pipe.vae.config.latent_channels * 4,
            packed_h,
            packed_w,
        )

        img_ids = pipe._prepare_latent_ids(latents)

        actual = self.cli(
            "Flux2KleinPipeline_embeddings",

            "--batch", str(batch),
            "--prompt", prompt,
            "--max_sequence_length", str(max_sequence_length),
            "--packed_h", str(packed_h),
            "--packed_w", str(packed_w),

            "--transformer-patch_size", "1",
            "--transformer-in_channels", "8",
            "--transformer-out_channels", "8",
            "--transformer-num_layers", "1",
            "--transformer-num_single_layers", "1",
            "--transformer-attention_head_dim", "4",
            "--transformer-num_attention_heads", "2",
            "--transformer-joint_attention_dim", "24",
            "--transformer-timestep_guidance_channels", "8",
            "--transformer-axes_dims_rope", "2",
            "--transformer-axes_dims_rope", "2",
            "--transformer-guidance_embeds", "false",

            "--vae-in_channels", "3",
            "--vae-out_channels", "3",
            "--vae-latent_channels", "4",
            "--vae-block_out_channels", "8",
            "--vae-layers_per_block", "1",
            "--vae-norm_num_groups", "8",

            "--text_encoder-hidden_size", "8",
            "--text_encoder-intermediate_size", "16",
            "--text_encoder-num_hidden_layers", "28",
            "--text_encoder-num_attention_heads", "2",
            "--text_encoder-num_key_value_heads", "2",
            "--text_encoder-max_position_embeddings", "32",

            "--tokenizer_file", self.tokenizer_file,

            *self.params(pipe.transformer, self.tmpdir.name, "transformer"),
            *self.params(pipe.vae, self.tmpdir.name, "vae"),
            *self.params(pipe.text_encoder, self.tmpdir.name, "text_encoder"),
        )

        self.assertTensors(actual, [prompt_embeds.float(), txt_ids.float(), img_ids.float()])
