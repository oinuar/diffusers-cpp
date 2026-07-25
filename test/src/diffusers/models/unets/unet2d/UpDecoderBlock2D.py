from utils import TestCase
import torch
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D

class TestNNUpDecoderBlock2D(TestCase):
    def test_no_upsample(self):
        model = UpDecoderBlock2D(
            num_layers=2,
            in_channels=8,
            out_channels=4,
            add_upsample=False,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
            temb_channels=None,
        )

        hidden_states = torch.randn(1, 8, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "UpDecoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "8",
            "--out_channels", "4",
            "--add_upsample", "false",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_upsample(self):
        model = UpDecoderBlock2D(
            num_layers=2,
            in_channels=8,
            out_channels=4,
            add_upsample=True,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
            temb_channels=None,
        )

        hidden_states = torch.randn(1, 8, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "UpDecoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "8",
            "--out_channels", "4",
            "--add_upsample", "true",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
