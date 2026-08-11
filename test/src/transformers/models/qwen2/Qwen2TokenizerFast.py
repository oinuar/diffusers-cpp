from utils import TestCase
from transformers import Qwen2TokenizerFast
import torch
import json
import pathlib
import os

class TestTransformersQwen2TokenizerFast(TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "utils" / "convert-model" / "tokenizer"

        if not tokenizer_dir.is_dir():
            raise RuntimeError("No such directory: {tokenizer_dir}. It is required for tokenizer-related tests.")

        cls.tokenizer_dir = str(tokenizer_dir)
        cls.tokenizer_file = str(tokenizer_dir / "tokenizer.json")

    def test_encode_word(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

        text = ""
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected = tokenizer.decode(ids, skip_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Decode",
            "--tokenizer_file", self.tokenizer_file,
            "--skip_special_tokens", "false",
            *sum([["--ids", str(id)] for id in ids], [])
        )

        self.assertEqual(actual[0].tolist(), [list(expected)])

    def test_decode_words(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
            enable_thinking=False
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            "--enable_thinking", "false",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_system_and_user(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
            enable_thinking=False,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            "--enable_thinking", "false",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_multi_turn(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

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
            enable_thinking=False,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_ApplyChatTemplate",
            "--tokenizer_file", self.tokenizer_file,
            "--add_generation_prompt", "true",
            "--enable_thinking", "false",
            *sum([["--messages", json.dumps(message)] for message in messages], [])
        )

        self.assertEqual(actual[0], [expected])

    def test_chat_message_tokens(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

        messages = [
            {
                "role": "user",
                "content": "hello world"
            }
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        expected = tokenizer.encode(text, add_special_tokens=False)

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--add_special_tokens", "false"
        )

        self.assertEqual(actual[0].to(torch.int32).tolist(), expected)

    def test_encode_with_padding_and_mask(self):
        tokenizer = Qwen2TokenizerFast.from_pretrained(self.tokenizer_dir)

        text = "hello world"

        max_length = 8

        expected = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )

        actual = self.cli(
            "Qwen2TokenizerFast_Encode",
            "--tokenizer_file", self.tokenizer_file,
            "--text", text,
            "--max_length", str(max_length),
            "--return_attention_mask", "true",
            "--add_special_tokens", "false",
        )

        self.assertEqual(
            actual[0].to(torch.int32).tolist(),
            expected["input_ids"],
        )

        self.assertEqual(
            actual[1].to(torch.int32).tolist(),
            expected["attention_mask"],
        )
