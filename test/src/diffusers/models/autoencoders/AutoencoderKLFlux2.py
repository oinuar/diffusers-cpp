from utils import TestCase
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

class TestNNAutoencoderKLFlux2(TestCase):
    def test_mode(self):
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        sample = torch.randn(1, 3, 16, 16)

        expected = model(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2",

            "--sample", str(sample.tolist()),

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",

            *self.params(model),
        )

        self.assertTensors(actual, [expected])
