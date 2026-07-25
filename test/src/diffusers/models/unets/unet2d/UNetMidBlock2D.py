from utils import TestCase
import torch
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D

class TestNN_UNetMidBlock2D(TestCase):
    def test_without_attention(self):
        model = UNetMidBlock2D(
            in_channels=8,
            attention_head_dim=8,
            resnet_groups=4,
            temb_channels=None,
            add_attention=False,
            resnet_act_fn="silu",
            resnet_time_scale_shift="default",
        )

        sample = torch.randn(
            1,
            8,
            4,
            4,
        )

        expected = model(sample)

        actual = self.cli(
            "UNetMidBlock2D",
            "--in_channels", "8",
            "--attention_head_dim", "8",
            "--resnet_groups", "4",
            "--add_attention", "false",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_attention(self):
        model = UNetMidBlock2D(
            in_channels=8,
            attention_head_dim=8,
            resnet_groups=4,
            temb_channels=None,
            resnet_act_fn="silu",
            resnet_time_scale_shift="default",
        )

        sample = torch.randn(
            1,
            8,
            4,
            4,
        )

        expected = model(sample)

        actual = self.cli(
            "UNetMidBlock2D",
            "--in_channels", "8",
            "--attention_head_dim", "8",
            "--resnet_groups", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
