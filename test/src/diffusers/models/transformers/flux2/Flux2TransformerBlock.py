import tempfile
from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2TransformerBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory(delete=False)

    def test_default(self):
        model = Flux2TransformerBlock(
            dim=80,
            num_attention_heads=2,
            attention_head_dim=40,
        )

        hidden_states = torch.randn(1, self.n_devices(), 80)
        encoder_hidden_states = torch.randn(1, self.n_devices(), 80)

        temb_mod_img = torch.randn(1, 1, 480)
        temb_mod_txt = torch.randn(1, 1, 480)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "80",
            "--num_attention_heads", "2",
            "--attention_head_dim", "40",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, list(expected))

    def test_bias(self):
        model = Flux2TransformerBlock(
            dim=80,
            num_attention_heads=2,
            attention_head_dim=40,
            bias=True,
        )

        hidden_states = torch.randn(1, self.n_devices(), 80)
        encoder_hidden_states = torch.randn(1, self.n_devices(), 80)

        temb_mod_img = torch.randn(1, 1, 480)
        temb_mod_txt = torch.randn(1, 1, 480)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "80",
            "--num_attention_heads", "2",
            "--attention_head_dim", "40",
            "--bias", "true",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, list(expected))

    def test_multiple_tokens(self):
        model = Flux2TransformerBlock(
            dim=80,
            num_attention_heads=2,
            attention_head_dim=40,
        )

        hidden_states = torch.randn(1, 4, 80)
        encoder_hidden_states = torch.randn(1, 3, 80)

        temb_mod_img = torch.randn(1, 1, 480)
        temb_mod_txt = torch.randn(1, 1, 480)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "80",
            "--num_attention_heads", "2",
            "--attention_head_dim", "40",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, list(expected))
