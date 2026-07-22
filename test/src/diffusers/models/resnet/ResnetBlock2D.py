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

            "--param-norm1-weight", str(model.norm1.weight.tolist()),
            "--param-norm1-bias", str(model.norm1.bias.tolist()),

            "--param-conv1-weight", str(model.conv1.weight.tolist()),
            "--param-conv1-bias", str(model.conv1.bias.tolist()),

            "--param-norm2-weight", str(model.norm2.weight.tolist()),
            "--param-norm2-bias", str(model.norm2.bias.tolist()),

            "--param-conv2-weight", str(model.conv2.weight.tolist()),
            "--param-conv2-bias", str(model.conv2.bias.tolist()),
        )

        self.assertTensors(actual, [expected], rtol=2e-4, atol=2e-5)

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

            "--param-norm1-weight", str(model.norm1.weight.tolist()),
            "--param-norm1-bias", str(model.norm1.bias.tolist()),

            "--param-conv1-weight", str(model.conv1.weight.tolist()),
            "--param-conv1-bias", str(model.conv1.bias.tolist()),

            "--param-norm2-weight", str(model.norm2.weight.tolist()),
            "--param-norm2-bias", str(model.norm2.bias.tolist()),

            "--param-conv2-weight", str(model.conv2.weight.tolist()),
            "--param-conv2-bias", str(model.conv2.bias.tolist()),

            "--param-conv_shortcut-weight", str(model.conv_shortcut.weight.tolist()),
            "--param-conv_shortcut-bias", str(model.conv_shortcut.bias.tolist()),
        )

        self.assertTensors(actual, [expected], rtol=2e-4, atol=2e-5)

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

            "--param-norm1-weight", str(model.norm1.weight.tolist()),
            "--param-norm1-bias", str(model.norm1.bias.tolist()),

            "--param-conv1-weight", str(model.conv1.weight.tolist()),
            "--param-conv1-bias", str(model.conv1.bias.tolist()),

            "--param-time_emb_proj-weight", str(model.time_emb_proj.weight.tolist()),
            "--param-time_emb_proj-bias", str(model.time_emb_proj.bias.tolist()),

            "--param-norm2-weight", str(model.norm2.weight.tolist()),
            "--param-norm2-bias", str(model.norm2.bias.tolist()),

            "--param-conv2-weight", str(model.conv2.weight.tolist()),
            "--param-conv2-bias", str(model.conv2.bias.tolist()),
        )

        self.assertTensors(actual, [expected], rtol=2e-4, atol=2e-5)