from utils import TestCase
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

class TestTransformersQwen3Model(TestCase):
    def test(self):
        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )

        model = Qwen3Model(config)

        input_ids = torch.tensor([[1, 2, 3]])

        expected = model(input_ids=input_ids).last_hidden_state

        actual = self.cli(
            "Qwen3Model",
            "--vocab_size", "16",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--num_hidden_layers", "1",
            "--num_attention_heads", "2",
            "--num_key_value_heads", "2",
            "--max_position_embeddings", "32",
            "--input_ids", str(input_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
