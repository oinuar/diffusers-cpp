from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2SingleTransformerBlock(TestCase):
    def test(self):
        model = Flux2SingleTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
        )

        hidden_states = torch.randn(1, 4, 8)
        temb_mod = torch.randn(1, 1, 24)  # 3 * dim

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=temb_mod,
        )

        actual = self.cli(
            "Flux2SingleTransformerBlock",
            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--temb_mod", str(temb_mod.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_encoder_hidden_states(self):
        model = Flux2SingleTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
        )

        hidden_states = torch.randn(1, 3, 8)
        encoder_hidden_states = torch.randn(1, 2, 8)
        temb_mod = torch.randn(1, 1, 24)

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod=temb_mod,
            split_hidden_states=True,
        )

        actual = self.cli(
            "Flux2SingleTransformerBlock",
            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod", str(temb_mod.tolist()),
            "--split_hidden_states", "true",
            *self.params(model),
        )

        self.assertTensors(actual, list(expected))
