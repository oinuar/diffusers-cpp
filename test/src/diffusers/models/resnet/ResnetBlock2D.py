from utils import TestCase
import torch
from diffusers.models.resnet import ResnetBlock2D


class TestNNResnetBlock2D(TestCase):
    def test_default(self):
        model = ResnetBlock2D(
            in_channels=8,
            out_channels=8,
            groups=8,
            temb_channels=None,
            non_linearity="silu"
        )

        hidden_states = torch.randn(1, 8, 2, 2)

        expected = model.forward(hidden_states, None)

        actual = self.cli(
            "ResnetBlock2D",
            "--in_channels", "8",
            "--out_channels", "8",
            "--groups", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_channel_projection(self):
        model = ResnetBlock2D(
            in_channels=8,
            out_channels=16,
            groups=8,
            temb_channels=None,
            non_linearity="silu"
        )

        hidden_states = torch.randn(1, 8, 2, 2)

        expected = model.forward(hidden_states, None)

        actual = self.cli(
            "ResnetBlock2D",
            "--in_channels", "8",
            "--out_channels", "16",
            "--groups", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_temb(self):
        model = ResnetBlock2D(
            in_channels=8,
            out_channels=8,
            groups=8,
            temb_channels=16,
            non_linearity="silu"
        )

        hidden_states = torch.randn(1, 8, 2, 2)
        temb = torch.randn(1, 16)

        expected = model.forward(hidden_states, temb)

        actual = self.cli(
            "ResnetBlock2D",
            "--in_channels", "8",
            "--out_channels", "8",
            "--groups", "8",
            "--temb_channels", "16",
            "--hidden_states", str(hidden_states.tolist()),
            "--temb", str(temb.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])