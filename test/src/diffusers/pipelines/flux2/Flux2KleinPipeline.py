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
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2TokenizerFast
from utils import TestCase

class TestPipelinesFlux2KleinPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<|endoftext|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 1,
                    "content": "<|im_start|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": 2,
                    "content": "<|im_end|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            ],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<|endoftext|>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": {
                    "<|endoftext|>": 0,
                    "<|im_start|>": 1,
                    "<|im_end|>": 2,
                    "h": 3,
                    "e": 4,
                    "l": 5,
                    "o": 6,
                    "Ġ": 7,
                    "he": 8,
                    "ll": 9,
                    "lo": 10,
                    "hel": 11,
                    "llo": 12,
                    "hello": 13,
                    "Ġhello": 14,
                    "world": 15,
                    "Ġworld": 16,
                },
                "merges": [
                    "h e",
                    "l l",
                    "l o",
                    "he l",
                    "l lo",
                    "hel lo",
                    "Ġ hello",
                    "Ġ world",
                ],
            },
        }

        tokenizer_config = {
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": 32768,
            "clean_up_tokenization_spaces": False,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "bos_token": None,
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
                "{% endif %}"
            ),
        }

        cls.tmpdir = tempfile.TemporaryDirectory(delete=False)

        json_path = os.path.join(cls.tmpdir.name, "tokenizer.json")
        config_path = os.path.join(cls.tmpdir.name, "tokenizer_config.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f)

        cls.tokenizer_file = json_path
        cls.tokenizer_config_file = config_path

    @classmethod
    def tearDownClass(cls):
        #cls.tmpdir.cleanup()
        pass

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
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )

        text_encoder = Qwen3ForCausalLM(text_config)

        tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.tmpdir.name
        )

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
            device="cpu",
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

            "--text_encoder-vocab_size", "17",
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
