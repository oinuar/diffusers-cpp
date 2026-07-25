from utils import TestCase
import torch
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D

class TestNNDownEncoderBlock2D(TestCase):
    def test_basic(self):
        model = DownEncoderBlock2D(
            num_layers=2,
            in_channels=4,
            out_channels=8,
            add_downsample=False,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
        )

        hidden_states = torch.randn(1, 4, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "DownEncoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "4",
            "--out_channels", "8",
            "--add_downsample", "false",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model)
        )

        self.assertTensors(actual, [expected])

    def test_with_downsample(self):
        model = DownEncoderBlock2D(
            num_layers=2,
            in_channels=4,
            out_channels=8,
            add_downsample=True,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
        )

        hidden_states = torch.randn(1, 4, 16, 16)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "DownEncoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "4",
            "--out_channels", "8",
            "--add_downsample", "true",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model)
        )

        self.assertTensors(actual, [expected])

    def test_with_downsample_padding_zero(self):
        model = DownEncoderBlock2D(
            num_layers=2,
            in_channels=8,
            out_channels=8,
            add_downsample=True,
            downsample_padding=0,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
        )

        hidden_states = torch.randn(1, 8, 16, 16)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "DownEncoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "8",
            "--out_channels", "8",
            "--add_downsample", "true",
            "--downsample_padding", "0",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model)
        )

        self.assertTensors(actual, [expected])