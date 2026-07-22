from utils import TestCase
import torch
from torch.nn.modules.normalization import GroupNorm

class TestNNNormalizationGroupNorm(TestCase):
    def test_default(self):
        model = GroupNorm(
            num_groups=2,
            num_channels=4,
            eps=1e-6,
        )

        input = torch.randn(2, 4, 3, 3)

        expected = model.forward(input)

        actual = self.cli(
            "GroupNorm",
            "--num_groups", "2",
            "--num_channels", "4",
            "--eps", "1e-6",
            "--input", str(input.tolist()),
            "--param-weight", str(model.weight.tolist()),
            "--param-bias", str(model.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_multiple_groups(self):
        model = GroupNorm(
            num_groups=4,
            num_channels=8,
            eps=1e-6,
        )

        input = torch.randn(1, 8, 4, 4)

        expected = model.forward(input)

        actual = self.cli(
            "GroupNorm",
            "--num_groups", "4",
            "--num_channels", "8",
            "--eps", "1e-6",
            "--input", str(input.tolist()),
            "--param-weight", str(model.weight.tolist()),
            "--param-bias", str(model.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_no_affine(self):
        model = GroupNorm(
            num_groups=2,
            num_channels=4,
            eps=1e-6,
            affine=False,
        )

        input = torch.randn(2, 4, 2, 2)

        expected = model.forward(input)

        actual = self.cli(
            "GroupNorm",
            "--num_groups", "2",
            "--num_channels", "4",
            "--eps", "1e-6",
            "--affine", "false",
            "--input", str(input.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_no_bias(self):
        model = GroupNorm(
            num_groups=2,
            num_channels=4,
            eps=1e-6,
            bias=False,
        )

        input = torch.randn(2, 4, 2, 2)

        expected = model.forward(input)

        actual = self.cli(
            "GroupNorm",
            "--num_groups", "2",
            "--num_channels", "4",
            "--eps", "1e-6",
            "--bias", "false",
            "--input", str(input.tolist()),
            "--param-weight", str(model.weight.tolist()),
        )

        self.assertTensors(actual, [expected])
