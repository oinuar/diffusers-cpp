from utils import TestCase
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP

class TestTransformersQwen3MLP(TestCase):
    def test(self):
        config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
        )

        model = Qwen3MLP(config)

        hidden_states = torch.randn(1, 8)

        expected = model(hidden_states)

        actual = self.cli(
            "Qwen3MLP",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_2x3x8(self):
        config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
        )

        model = Qwen3MLP(config)

        hidden_states = torch.randn(2, 3, 8)

        expected = model(hidden_states)

        actual = self.cli(
            "Qwen3MLP",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_small_intermediate(self):
        config = Qwen3Config(
            hidden_size=4,
            intermediate_size=6,
            hidden_act="silu",
        )

        model = Qwen3MLP(config)

        hidden_states = torch.randn(1, 4)

        expected = model(hidden_states)

        actual = self.cli(
            "Qwen3MLP",
            "--hidden_size", "4",
            "--intermediate_size", "6",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])