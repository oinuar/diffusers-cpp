from utils import TestCase
import torch
from diffusers.models.embeddings import Timesteps

class TestNNTimesteps(TestCase):
    def test_default(self):
        model = Timesteps(
            num_channels=8,
            flip_sin_to_cos=False,
            downscale_freq_shift=1,
        )

        timesteps = torch.tensor([0.0, 1.0])

        expected = model(timesteps)

        actual = self.cli(
            "Timesteps",
            "--num_channels", "8",
            "--flip_sin_to_cos", "false",
            "--downscale_freq_shift", "1.0",
            "--timesteps", str(timesteps.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_flip(self):
        model = Timesteps(
            num_channels=8,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )

        timesteps = torch.tensor([5.0])

        expected = model(timesteps)

        actual = self.cli(
            "Timesteps",
            "--num_channels", "8",
            "--flip_sin_to_cos", "true",
            "--downscale_freq_shift", "1.0",
            "--timesteps", str(timesteps.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_scale_and_shift(self):
        model = Timesteps(
            num_channels=8,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.5,
            scale=2.0,
        )

        timesteps = torch.tensor([2.0])

        expected = model(timesteps)

        actual = self.cli(
            "Timesteps",
            "--num_channels", "8",
            "--flip_sin_to_cos", "false",
            "--downscale_freq_shift", "0.5",
            "--scale", "2.0",
            "--timesteps", str(timesteps.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_odd_channels(self):
        model = Timesteps(
            num_channels=9,
            flip_sin_to_cos=False,
            downscale_freq_shift=1,
        )

        timesteps = torch.tensor([3.0])

        expected = model(timesteps)

        actual = self.cli(
            "Timesteps",
            "--num_channels", "9",
            "--flip_sin_to_cos", "false",
            "--downscale_freq_shift", "1.0",
            "--timesteps", str(timesteps.tolist()),
        )

        self.assertTensors(actual, [expected])