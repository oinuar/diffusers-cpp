import json
import os
import tempfile
import torch
from diffusers import (
    AutoencoderKLFlux2,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2TokenizerFast
import numpy as np
from PIL import Image
from utils import TestCase

class TestPipelinesFlux2KleinPipeline(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "transformers", "models", "qwen2")
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

            "--tokenizer_dir", self.tokenizer_dir,

            *self.params(pipe.transformer, self.tmpdir.name, "transformer"),
            *self.params(pipe.vae, self.tmpdir.name, "vae"),
            *self.params(pipe.text_encoder, self.tmpdir.name, "text_encoder"),
        )

        self.assertTensors(actual, [prompt_embeds.float(), txt_ids.float(), img_ids.float()])

    def test_denoise(self):
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=16,
            out_channels=16,
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
        generator = torch.Generator()

        packed_h = 2
        packed_w = 3

        # ------------------------------------------------------------
        # 1. Prompt embeddings
        # ------------------------------------------------------------

        prompt_embeds, txt_ids = pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=batch,
            max_sequence_length=max_sequence_length,
        )

        # ------------------------------------------------------------
        # 2. Initial latents + latent IDs
        #
        # Generate the latent explicitly so Python and C++ operate on
        # exactly the same tensor.
        # ------------------------------------------------------------

        init_latents_unpacked = torch.randn(
            batch,
            pipe.vae.config.latent_channels * 4,
            packed_h,
            packed_w,
            generator=generator,
            dtype=prompt_embeds.dtype,
        )

        init_latents, img_ids = pipe.prepare_latents(
            batch_size=batch,
            num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=packed_h * 2 * pipe.vae_scale_factor,
            width=packed_w * 2 * pipe.vae_scale_factor,
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
            generator=generator,
            latents=init_latents_unpacked,
        )

        # ------------------------------------------------------------
        # 4. Scheduler setup
        # ------------------------------------------------------------

        num_inference_steps = 1
        image_seq_len = packed_h * packed_w

        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )

        sigmas = np.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
        )

        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )

        # Exactly the timestep used by the first iteration of __call__.
        timestep = timesteps[0]

        # Time delta
        sigma = scheduler.sigmas[0]
        next_sigma = scheduler.sigmas[1]
        dt = next_sigma - sigma

        # ------------------------------------------------------------
        # 5. Transformer
        #
        # This mirrors the actual Klein denoising loop:
        #
        #   timestep / 1000
        #   guidance=None
        #   hidden_states=latents
        #   encoder_hidden_states=prompt_embeds
        #   img_ids=img_ids
        #   txt_ids=txt_ids
        # ------------------------------------------------------------

        timestep_input = timestep.expand(batch)

        noise_pred = pipe.transformer(
            hidden_states=init_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_input / 1000.0,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]

        # ------------------------------------------------------------
        # 6. Actual scheduler transition
        # ------------------------------------------------------------

        expected_latents = pipe.scheduler.step(
            noise_pred,
            timestep,
            init_latents,
            return_dict=False,
        )[0]

        actual = self.cli(
            "Flux2KleinPipeline_denoise",

            "--batch", str(batch),
            "--packed_h", str(packed_h),
            "--packed_w", str(packed_w),
            "--num_ref_tokens", "0",
            "--timestep", str(timestep.tolist()),
            "--dt", str(dt.tolist()),

            "--prompt_embeds", str(prompt_embeds.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--init_latents", str(init_latents.tolist()),

            "--transformer-patch_size", "1",
            "--transformer-in_channels", "16",
            "--transformer-out_channels", "16",
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

            "--tokenizer_dir", self.tokenizer_dir,

            *self.params(pipe.transformer, self.tmpdir.name, "transformer"),
            *self.params(pipe.vae, self.tmpdir.name, "vae"),
            *self.params(pipe.text_encoder, self.tmpdir.name, "text_encoder"),
        )

        self.assertTensors(actual, [expected_latents.float()])

    def test_call(self):
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=16,
            out_channels=16,
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
        num_inference_steps = 2
        generator = torch.Generator()

        packed_h = 2
        packed_w = 3

        height = packed_h * 2 * pipe.vae_scale_factor
        width = packed_w * 2 * pipe.vae_scale_factor

        # ------------------------------------------------------------
        # 1. Initial latents
        #
        # Generated in Python and passed to the C++ CLI so that both
        # sides run the complete denoising loop on exactly the same
        # input.
        # ------------------------------------------------------------

        init_latents = torch.randn(
            batch,
            pipe.vae.config.latent_channels * 4,
            packed_h,
            packed_w,
            generator=generator,
            dtype=torch.float32,
        )

        # ------------------------------------------------------------
        # 2. Python reference: complete __call__ with 2 denoising steps.
        #
        # guidance_scale=1.0 disables classifier-free guidance, matching
        # the (guidance-distilled) C++ pipeline.
        # ------------------------------------------------------------

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            max_sequence_length=max_sequence_length,
            latents=init_latents,
        )

        expected = [
            torch.from_numpy(np.array(image)).float()
            for image in output.images
        ]

        actual = self.cli(
            "Flux2KleinPipeline_call",

            "--prompt", prompt,
            "--height", str(height),
            "--width", str(width),
            "--num_inference_steps", str(num_inference_steps),
            "--max_sequence_length", str(max_sequence_length),
            "--init_latents", str(pipe._pack_latents(init_latents).tolist()),

            "--transformer-patch_size", "1",
            "--transformer-in_channels", "16",
            "--transformer-out_channels", "16",
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

            "--tokenizer_dir", self.tokenizer_dir,

            *self.params(pipe.transformer, self.tmpdir.name, "transformer"),
            *self.params(pipe.vae, self.tmpdir.name, "vae"),
            *self.params(pipe.text_encoder, self.tmpdir.name, "text_encoder"),
        )

        self.assertTensors(actual, expected)
