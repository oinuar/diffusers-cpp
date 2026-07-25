from utils import TestCase
import torch
from diffusers.models.autoencoders.vae import Decoder

class TestNNDecoder(TestCase):
    def test_default(self):
        model = Decoder(
            in_channels=4,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
        )

        sample = torch.randn(1, 4, 8, 8)

        expected = model(sample)

        actual = self.cli(
            "Decoder",
            "--in_channels", "4",
            "--out_channels", "3",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_multiple_blocks(self):
        model = Decoder(
            in_channels=4,
            out_channels=3,
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(8, 16),
            layers_per_block=1,
            norm_num_groups=4,
        )

        sample = torch.randn(1, 4, 8, 8)

        expected = model(sample)

        actual = self.cli(
            "Decoder",
            "--in_channels", "4",
            "--out_channels", "3",
            "--block_out_channels", "8",
            "--block_out_channels", "16",
            "--layers_per_block", "1",
            "--norm_num_groups", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
