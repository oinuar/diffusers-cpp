from utils import TestCase
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

class TestNNQwen3ForCausalLM(TestCase):
    def test_tiny(self):
        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )

        model = Qwen3ForCausalLM(config)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        expected = model(input_ids=input_ids).logits

        actual = self.cli(
            "Qwen3ForCausalLM",
            "--vocab_size", "16",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--num_hidden_layers", "1",
            "--num_attention_heads", "2",
            "--num_key_value_heads", "2",
            "--max_position_embeddings", "16",
            "--input_ids", str(input_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_batch(self):
        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )

        model = Qwen3ForCausalLM(config)

        input_ids = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=torch.long,
        )

        expected = model(input_ids=input_ids).logits

        actual = self.cli(
            "Qwen3ForCausalLM",
            "--vocab_size", "16",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--num_hidden_layers", "1",
            "--num_attention_heads", "2",
            "--num_key_value_heads", "2",
            "--max_position_embeddings", "16",
            "--input_ids", str(input_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_two_layers(self):
        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )

        model = Qwen3ForCausalLM(config)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        expected = model(input_ids=input_ids).logits

        actual = self.cli(
            "Qwen3ForCausalLM",
            "--vocab_size", "16",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--num_hidden_layers", "2",
            "--num_attention_heads", "2",
            "--num_key_value_heads", "2",
            "--max_position_embeddings", "16",
            "--input_ids", str(input_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
