import tempfile
from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2SingleTransformerBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory(delete=False)

    def test(self):
        model = Flux2SingleTransformerBlock(
            dim=80,
            num_attention_heads=2,
            attention_head_dim=40,
        )

        hidden_states = torch.randn(1, 4, 80)
        temb_mod = torch.randn(1, 1, 240)  # 3 * dim

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=temb_mod,
        )

        actual = self.cli(
            "Flux2SingleTransformerBlock",
            "--dim", "80",
            "--num_attention_heads", "2",
            "--attention_head_dim", "40",
            "--hidden_states", str(hidden_states.tolist()),
            "--temb_mod", str(temb_mod.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, [expected])

    def test_with_encoder_hidden_states(self):
        model = Flux2SingleTransformerBlock(
            dim=80,
            num_attention_heads=2,
            attention_head_dim=40,
        )

        hidden_states = torch.randn(1, 3, 80)
        encoder_hidden_states = torch.randn(1, 2, 80)
        temb_mod = torch.randn(1, 1, 240)

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod=temb_mod,
            split_hidden_states=True,
        )

        actual = self.cli(
            "Flux2SingleTransformerBlock",
            "--dim", "80",
            "--num_attention_heads", "2",
            "--attention_head_dim", "40",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod", str(temb_mod.tolist()),
            "--split_hidden_states", "true",
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, list(expected))
