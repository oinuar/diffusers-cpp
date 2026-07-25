from utils import TestCase
import torch
from diffusers.models.downsampling import Downsample2D

class TestNNDownsample2D(TestCase):
    def test_without_conv(self):
        model = Downsample2D(
            channels=4
        )

        hidden_states = torch.randn(1, 4, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Downsample2D",
            "--channels", "4",
            "--hidden_states", str(hidden_states.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_with_conv(self):
        model = Downsample2D(
            channels=4,
            use_conv=True,
        )

        hidden_states = torch.randn(1, 4, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Downsample2D",
            "--channels", "4",
            "--use_conv", "true",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-conv-weight", str(model.conv.weight.tolist()),
            "--param-conv-bias", str(model.conv.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_with_conv_non_square(self):
        model = Downsample2D(
            channels=4,
            use_conv=True,
        )

        hidden_states = torch.randn(2, 4, 7, 9)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Downsample2D",
            "--channels", "4",
            "--use_conv", "true",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-conv-weight", str(model.conv.weight.tolist()),
            "--param-conv-bias", str(model.conv.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_with_conv_padding_zero(self):
        model = Downsample2D(
            channels=8,
            use_conv=True,
            out_channels=8,
            padding=0,
        )

        hidden_states = torch.randn(1, 8, 16, 16)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Downsample2D",
            "--channels", "8",
            "--out_channels", "8",
            "--use_conv", "true",
            "--padding", "0",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-conv-weight", str(model.conv.weight.tolist()),
            "--param-conv-bias", str(model.conv.bias.tolist()),
        )

        self.assertTensors(actual, [expected])
