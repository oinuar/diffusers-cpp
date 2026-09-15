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
        cls.tmpdir = tempfile.TemporaryDirectory(delete=False)

    def make_ref_image(self, width, height, generator):
        # Deterministic 8-bit RGB reference image. Saved as PNG (lossless) so
        # the C++ CLI (Image::from_file) sees exactly these pixels.
        pixels = torch.randint(0, 256, (height, width, 3), generator=generator).to(torch.uint8)
        image = Image.fromarray(pixels.numpy(), mode="RGB")

        path = os.path.join(self.tmpdir.name, f"ref-{width}x{height}.png")
        image.save(path)

        return image, path

    def prepare_condition_images(self, pipe, images):
        # Mirrors the reference image preprocessing in
        # Flux2KleinPipeline.__call__.
        condition_images = []

        for img in images:
            image_width, image_height = img.size

            if image_width * image_height > 1024 * 1024:
                img = pipe.image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size

            multiple_of = pipe.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of

            img = pipe.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_images.append(img)

        return condition_images

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

    def test_embeddings_with_ref_images(self):
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
        generator = torch.Generator()

        # Reference image sizes are exact multiples of vae_scale_factor * 2,
        # so the resize+crop preprocessing keeps the pixels untouched on both
        # the Python and the C++ side.
        ref_images = []
        ref_paths = []
        for width, height in [(8, 12), (12, 8)]:
            image, path = self.make_ref_image(width, height, generator)
            ref_images.append(image)
            ref_paths.append(path)

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

        condition_images = self.prepare_condition_images(pipe, ref_images)

        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=condition_images,
            batch_size=batch,
            generator=generator,
            device=prompt_embeds.device,
            dtype=pipe.vae.dtype,
        )

        actual = self.cli(
            "Flux2KleinPipeline_embeddings",

            "--batch", str(batch),
            "--prompt", prompt,
            "--max_sequence_length", str(max_sequence_length),
            "--packed_h", str(packed_h),
            "--packed_w", str(packed_w),
            *sum([["--images", path] for path in ref_paths], []),

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

        self.assertTensors(actual, [
            prompt_embeds.float(),
            txt_ids.float(),
            img_ids.float(),
            image_latents.float(),
            image_latent_ids.float(),
        ])

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

    def test_denoise_with_ref_images(self):
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
        # 3. Reference image latents + IDs
        #
        # The reference image size is an exact multiple of
        # vae_scale_factor * 2, so the resize+crop preprocessing keeps
        # the pixels untouched on both the Python and the C++ side.
        # ------------------------------------------------------------

        ref_image, _ = self.make_ref_image(8, 12, generator)

        condition_images = self.prepare_condition_images(pipe, [ref_image])

        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=condition_images,
            batch_size=batch,
            generator=torch.Generator(),
            device=prompt_embeds.device,
            dtype=pipe.vae.dtype,
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
        # 5. Transformer with reference tokens
        #
        # This mirrors the actual Klein denoising loop:
        #
        #   latent_model_input = cat([latents, image_latents], dim=1)
        #   latent_image_ids   = cat([latent_ids, image_latent_ids], dim=1)
        #   timestep / 1000
        #   guidance=None
        #   noise_pred = noise_pred[:, :latents.size(1):]
        # ------------------------------------------------------------

        latent_model_input = torch.cat([init_latents, image_latents], dim=1)
        latent_image_ids = torch.cat([img_ids, image_latent_ids], dim=1)

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep.expand(batch) / 1000.0,
            img_ids=latent_image_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]

        noise_pred = noise_pred[:, :init_latents.size(1):]

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
            "--num_ref_tokens", str(image_latents.shape[1]),
            "--timestep", str(timestep.tolist()),
            "--dt", str(dt.tolist()),

            "--prompt_embeds", str(prompt_embeds.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--init_latents", str(init_latents.tolist()),
            "--image_latents", str(image_latents.tolist()),
            "--image_latent_ids", str(image_latent_ids.tolist()),

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

    def test_decode(self):
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
        packed_h = 2
        packed_w = 3

        latent_height = packed_h * 2
        latent_width = packed_w * 2

        # ------------------------------------------------------------
        # 1. Generate the same latent representation that the pipeline
        #    passes into the denoising stage.
        # ------------------------------------------------------------

        generator = torch.Generator()

        # The pipeline's packed latent representation is:
        #
        #   (B, N, 4C)
        #
        # where N = packed_h * packed_w.
        #
        # Start from the same representation used by prepare_latents().
        latents_unpacked = torch.randn(
            batch,
            pipe.vae.config.latent_channels * 4,
            packed_h,
            packed_w,
            generator=generator,
            dtype=torch.float32,
        )

        # Construct the latent IDs in the same way as the pipeline.
        latent_ids = pipe._prepare_latent_ids(latents_unpacked)

        # Convert to the packed representation consumed by the transformer.
        latents, _ = pipe.prepare_latents(
            batch_size=batch,
            num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=latent_height * pipe.vae_scale_factor,
            width=latent_width * pipe.vae_scale_factor,
            dtype=torch.float32,
            device=latents_unpacked.device,
            generator=generator,
            latents=latents_unpacked,
        )

        # ------------------------------------------------------------
        # 2. Reference Python decode path
        #
        # Flux2Klein's decode path first converts the packed latent tokens
        # back to spatial latents using _unpack_latents_with_ids().
        # ------------------------------------------------------------

        z = pipe._unpack_latents_with_ids(
            latents,
            latent_ids,
            latent_height // 2,
            latent_width // 2,
        )

        # ------------------------------------------------------------
        # 3. Apply the unnormalization.
        # ------------------------------------------------------------

        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            z.device, z.dtype
        )
        z = z * latents_bn_std + latents_bn_mean
        z = pipe._unpatchify_latents(z)

        # ------------------------------------------------------------
        # 4. Python VAE decode.
        # ------------------------------------------------------------

        expected = pipe.vae.decode(z, return_dict=False)[0]

        actual = self.cli(
            "Flux2KleinPipeline_decode",

            "--batch", str(batch),
            "--packed_h", str(packed_h),
            "--packed_w", str(packed_w),
            "--latents", str(latents.tolist()),

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

        self.assertTensors(actual, [expected.float()])

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

    def test_pack_latents(self):
        generator = torch.Generator()
        latents = torch.randn(2, 16, 2, 3, generator=generator)

        expected = Flux2KleinPipeline._pack_latents(latents)

        actual = self.cli(
            "Flux2KleinPipeline_pack_latents",
            "--latents", str(latents.tolist()),
        )

        self.assertTensors(actual, [expected.float()])

    def test_unpack_latents(self):
        generator = torch.Generator()
        packed = torch.randn(2, 2 * 3, 16, generator=generator)

        # Canonical grid ids, the ones the pipeline generates for the
        # generated latents via _prepare_latent_ids().
        x_ids = Flux2KleinPipeline._prepare_latent_ids(torch.zeros(2, 16, 2, 3))

        expected = Flux2KleinPipeline._unpack_latents_with_ids(packed, x_ids, 2, 3)

        actual = self.cli(
            "Flux2KleinPipeline_unpack_latents",
            "--latents", str(packed.tolist()),
            "--packed_h", "2",
            "--packed_w", "3",
        )

        self.assertTensors(actual, [expected.float()])

    def test_patchify_latents(self):
        generator = torch.Generator()
        latents = torch.randn(2, 4, 4, 6, generator=generator)

        expected = Flux2KleinPipeline._patchify_latents(latents)

        actual = self.cli(
            "Flux2KleinPipeline_patchify_latents",
            "--latents", str(latents.tolist()),
            "--channels", "4",
            "--packed_h", "2",
            "--packed_w", "3",
        )

        self.assertTensors(actual, [expected.float()])

    def test_unpatchify_latents(self):
        generator = torch.Generator()
        latents = torch.randn(2, 16, 2, 3, generator=generator)

        expected = Flux2KleinPipeline._unpatchify_latents(latents)

        actual = self.cli(
            "Flux2KleinPipeline_unpatchify_latents",
            "--latents", str(latents.tolist()),
            "--channels", "4",
            "--packed_h", "2",
            "--packed_w", "3",
        )

        self.assertTensors(actual, [expected.float()])

