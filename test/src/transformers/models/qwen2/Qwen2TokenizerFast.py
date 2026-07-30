from utils import TestCase
from transformers import Qwen2TokenizerFast
import torch
import json
import tempfile
import os


class TestTransformersQwen2TokenizerFast(TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Minimal tokenizer.json
        # Note: BBPE requires spaces to be represented as 'Ġ' (U+0120) in the vocab/merges
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalizer": None, "special": True},
                {"id": 1, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalizer": None, "special": True},
                {"id": 2, "content": "<|im_end|>", "single_word": False, "lstrip": False, "rstrip": False, "normalizer": None, "special": True}
            ],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True
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
                    "Ġ": 7,       # BBPE space character
                    "he": 8,
                    "ll": 9,
                    "lo": 10,
                    "hel": 11,
                    "llo": 12,
                    "hello": 13,
                    "Ġhello": 14,  # Space + hello
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
                    "Ġ world"
                ]
            }
        }

        # 2. Minimal tokenizer_config.json
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
                "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
        }

        cls.tmpdir = tempfile.TemporaryDirectory()

        tmpdir = cls.tmpdir.name

        json_path = os.path.join(tmpdir, "tokenizer.json")
        config_path = os.path.join(tmpdir, "tokenizer_config.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f)

        cls.tokenizer_file = json_path
        cls.tokenizer_config = config_path

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_encode_word(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "hello"
        expected = tokenizer.encode(text, add_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--add_special_tokens", "false"
        )

        self.assertEqual(actual[0].to(torch.int32).tolist(), expected)

    def test_encode_empty(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = ""
        expected = tokenizer.encode(text, add_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--add_special_tokens", "false"
        )

        self.assertEqual(actual[0].to(torch.int32).tolist(), expected)

    def test_encode_words(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "hello world"
        expected = tokenizer.encode(text, add_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--add_special_tokens", "false"
        )

        self.assertEqual(actual[0].to(torch.int32).tolist(), expected)

    def test_encode_special(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "<|endoftext|>"
        expected = tokenizer.encode(text, add_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--add_special_tokens", "false"
        )

        self.assertEqual(actual[0].to(torch.int32).tolist(), expected)

    def test_decode_word(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "hello"
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Decode",
            "--tokenizer_file", self.tokenizer_file,
            "--skip_special_tokens", "false",
            *sum([["--ids", str(id)] for id in ids], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_decode_empty(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = ""
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Decode",
            "--tokenizer_file", self.tokenizer_file,
            "--skip_special_tokens", "false",
            *sum([["--ids", str(id)] for id in ids], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_decode_words(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "hello world"
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Decode",
            "--tokenizer_file", self.tokenizer_file,
            "--skip_special_tokens", "false",
            *sum([["--ids", str(id)] for id in ids], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_decode_special(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        text = "<|endoftext|>"
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Decode",
            "--tokenizer_file", self.tokenizer_file,
            "--skip_special_tokens", "false",
            *sum([["--ids", str(id)] for id in ids], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_single_user_message(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        messages = [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
        
        expected = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_system_and_user(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello"
            }
        ]
        
        expected = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_multi_turn(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tmpdir.name)

        messages = [
            {
                "role": "user",
                "content": "What is 2+2?"
            },
            {
                "role": "assistant",
                "content": "4"
            },
            {
                "role": "user",
                "content": "Explain."
            }
        ]
        
        expected = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])
