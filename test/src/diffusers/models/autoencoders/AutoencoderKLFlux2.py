from utils import TestCase
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2


def make_vae():
    # A single-block VAE: the scale factor is 1, so latent sizes equal
    # pixel sizes and small inputs span multiple tiles.
    return AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        norm_num_groups=8,
    )


def vae_config_args():
    return [
        "--in_channels", "3",
        "--out_channels", "3",
        "--latent_channels", "4",
        "--block_out_channels", "8",
        "--layers_per_block", "1",
        "--norm_num_groups", "8",
    ]
class TestNNAutoencoderKLFlux2(TestCase):

    def test_mode(self):
        model = make_vae()

        sample = torch.randn(1, 3, 16, 16)

        expected = model(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2",

            "--sample", str(sample.tolist()),

            *vae_config_args(),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_vae_encode_tiled(self):
        # The tiled encode must match the reference tiled encode
        # (diffusers' _tiled_encode with the same tile size; the latent
        # mode is the first half of the returned moments).
        model = make_vae()

        # 48x32 image with 24x24 tiles (18 stride): a 3x3 tile grid with
        # clipped right and bottom borders, so the tiling and the tile
        # blending are exercised in both dimensions.
        sample = torch.randn(1, 3, 48, 32)

        model.tile_sample_min_size = 24
        model.tile_latent_min_size = 24
        model.use_tiling = True

        expected = model._tiled_encode(sample)[:, :model.config.latent_channels]

        actual = self.cli(
            "AutoencoderKLFlux2_encode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "24",

            *vae_config_args(),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_vae_decode_tiled(self):
        # The tiled decode must match the reference tiled decode
        # (diffusers' tiled_decode with the same tile size).
        model = make_vae()

        # 36x32 latent with 16x16 tiles (12 stride): a 3x3 tile grid with
        # clipped right and bottom borders, so the tiling and the tile
        # blending are exercised in both dimensions.
        sample = torch.randn(1, model.config.latent_channels, 36, 32)

        model.tile_sample_min_size = 16
        model.tile_latent_min_size = 16
        model.use_tiling = True

        expected = model.tiled_decode(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2_decode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "16",

            *vae_config_args(),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
