from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

class TestFlux2Modulation(TestCase):
    def test(self):
        model = Flux2Modulation(
            dim=8,
        )

        temb = torch.randn(1, 8)

        expected = model(temb)

        actual = self.cli(
            "Flux2Modulation",
            "--dim", "8",
            "--temb", str(temb.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_bias(self):
        model = Flux2Modulation(
            dim=8,
            bias=True
        )

        temb = torch.randn(1, 8)

        expected = model(temb)

        actual = self.cli(
            "Flux2Modulation",
            "--dim", "8",
            "--bias", "true",
            "--temb", str(temb.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
