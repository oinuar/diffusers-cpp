from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward

class TestFlux2FeedForward(TestCase):
    def test(self):
        model = Flux2FeedForward(
            dim=8,
            dim_out=8,
        )

        x = torch.randn(2, 8)

        expected = model(x)

        actual = self.cli(
            "Flux2FeedForward",
            "--dim", "8",
            "--dim_out", "8",
            "--x", str(x.tolist()),
            "--param-linear_in-weight", str(model.linear_in.weight.tolist()),
            "--param-linear_out-weight", str(model.linear_out.weight.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_dim_out(self):
        model = Flux2FeedForward(
            dim=8,
            dim_out=12,
        )

        x = torch.randn(3, 8)

        expected = model(x)

        actual = self.cli(
            "Flux2FeedForward",
            "--dim", "8",
            "--dim_out", "12",
            "--x", str(x.tolist()),
            "--param-linear_in-weight", str(model.linear_in.weight.tolist()),
            "--param-linear_out-weight", str(model.linear_out.weight.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_bias(self):
        model = Flux2FeedForward(
            dim=8,
            dim_out=8,
            bias=True
        )

        x = torch.randn(2, 8)

        expected = model(x)

        actual = self.cli(
            "Flux2FeedForward",
            "--dim", "8",
            "--dim_out", "8",
            "--bias", "true",
            "--x", str(x.tolist()),
            "--param-linear_in-weight", str(model.linear_in.weight.tolist()),
            "--param-linear_in-bias", str(model.linear_in.bias.tolist()),
            "--param-linear_out-weight", str(model.linear_out.weight.tolist()),
            "--param-linear_out-bias", str(model.linear_out.bias.tolist()),
        )

        self.assertTensors(actual, [expected])
