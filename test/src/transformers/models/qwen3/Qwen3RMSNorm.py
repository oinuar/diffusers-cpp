from utils import TestCase
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

class TestTransformersQwen3RMSNorm(TestCase):
    def test(self):
        model = Qwen3RMSNorm(8)

        hidden_states = torch.randn(2, 8)

        expected = model(hidden_states)

        actual = self.cli(
            "Qwen3RMSNorm",
            "--hidden_size", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_2x3x8(self):
        model = Qwen3RMSNorm(8)

        hidden_states = torch.randn(2, 3, 8)

        expected = model(hidden_states)

        actual = self.cli(
            "Qwen3RMSNorm",
            "--hidden_size", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
