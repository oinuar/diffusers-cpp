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
            "--param-norm_layer-weight", str(model.norm_layer.weight.tolist()),
            "--param-norm_layer-bias", str(model.norm_layer.bias.tolist()),
            "--param-conv_y-weight", str(model.conv_y.weight.tolist()),
            "--param-conv_y-bias", str(model.conv_y.bias.tolist()),
            "--param-conv_b-weight", str(model.conv_b.weight.tolist()),
            "--param-conv_b-bias", str(model.conv_b.bias.tolist()),
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
            "--param-norm_layer-weight", str(model.norm_layer.weight.tolist()),
            "--param-norm_layer-bias", str(model.norm_layer.bias.tolist()),
            "--param-conv_y-weight", str(model.conv_y.weight.tolist()),
            "--param-conv_y-bias", str(model.conv_y.bias.tolist()),
            "--param-conv_b-weight", str(model.conv_b.weight.tolist()),
            "--param-conv_b-bias", str(model.conv_b.bias.tolist()),
        )

        self.assertTensors(actual, [expected])