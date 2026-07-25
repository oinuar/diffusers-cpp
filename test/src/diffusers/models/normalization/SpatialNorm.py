from utils import TestCase
import torch
from diffusers.models.attention_processor import SpatialNorm

class TestNNSpatialNorm(TestCase):
    def test_basic(self):
        model = SpatialNorm(
            f_channels=32,
            zq_channels=4,
        )

        f = torch.randn(1, 32, 4, 4)
        zq = torch.randn(1, 4, 2, 2)

        expected = model.forward(f, zq)

        actual = self.cli(
            "SpatialNorm",
            "--f_channels", "4",
            "--zq_channels", "3",
            "--f", str(f.tolist()),
            "--zq", str(zq.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_same_resolution(self):
        model = SpatialNorm(
            f_channels=32,
            zq_channels=4,
        )

        f = torch.randn(2, 32, 8, 8)
        zq = torch.randn(2, 4, 8, 8)

        expected = model.forward(f, zq)

        actual = self.cli(
            "SpatialNorm",
            "--f_channels", "4",
            "--zq_channels", "3",
            "--f", str(f.tolist()),
            "--zq", str(zq.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])