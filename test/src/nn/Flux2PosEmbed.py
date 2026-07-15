from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

class TestNNFlux2PosEmbed(TestCase):
    def test(self):
        model = Flux2PosEmbed(
            theta=10000,
            axes_dim=[4, 4],
        )

        ids = torch.tensor([
            [0, 1],
            [2, 3],
        ])

        expected_cos, expected_sin = model(ids)

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "4",
            "--axes_dim", "4",
            "--ids", str(ids.tolist()),
        )

        self.assertTensors(
            actual,
            [expected_cos, expected_sin],
        )

    def test_mixed_axes(self):
        model = Flux2PosEmbed(
            theta=10000,
            axes_dim=[2, 6],
        )

        ids = torch.tensor([
            [5, 7],
        ])

        expected_cos, expected_sin = model(ids)

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "2",
            "--axes_dim", "6",
            "--ids", str(ids.tolist()),
        )

        self.assertTensors(
            actual,
            [expected_cos, expected_sin],
        )
