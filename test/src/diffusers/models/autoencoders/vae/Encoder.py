from utils import TestCase
import torch
from diffusers.models.autoencoders.vae import Encoder

class TestNNEncoder(TestCase):
    def test_default(self):
        model = Encoder(
            in_channels=3,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
        )

        sample = torch.randn(1, 3, 8, 8)

        expected = model(sample)

        actual = self.cli(
            "Encoder",
            "--in_channels", "3",
            "--out_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_multiple_blocks(self):
        model = Encoder(
            in_channels=3,
            out_channels=4,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            block_out_channels=(8, 16),
            layers_per_block=1,
            norm_num_groups=4,
        )

        sample = torch.randn(1, 3, 16, 16)

        expected = model(sample)

        actual = self.cli(
            "Encoder",
            "--in_channels", "3",
            "--out_channels", "4",
            "--block_out_channels", "8",
            "--block_out_channels", "16",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_no_double_z(self):
        model = Encoder(
            in_channels=3,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
            double_z=False,
        )

        sample = torch.randn(1, 3, 8, 8)

        expected = model(sample)

        actual = self.cli(
            "Encoder",
            "--in_channels", "3",
            "--out_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--double_z", "false",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_no_mid_attention(self):
        model = Encoder(
            in_channels=3,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
            mid_block_add_attention=False,
        )

        sample = torch.randn(1, 3, 8, 8)

        expected = model(sample)

        actual = self.cli(
            "Encoder",
            "--in_channels", "3",
            "--out_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--mid_block_add_attention", "false",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])