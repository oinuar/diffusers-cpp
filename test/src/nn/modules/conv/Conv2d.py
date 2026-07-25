from utils import TestCase
import torch
from torch.nn.modules.conv import Conv2d

class TestNNConv2d(TestCase):
    def test_3x4_kernel3(self):
        model = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        x = torch.randn(1, 3, 8, 8)

        expected = model(x)

        actual = self.cli(
            "Conv2d",
            "--in_channels", "3",
            "--out_channels", "4",
            "--kernel_size", "3",
            "--stride", "1",
            "--padding", "1",
            "--x", str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_stride2(self):
        model = Conv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        x = torch.randn(2, 3, 16, 16)

        expected = model(x)

        actual = self.cli(
            "Conv2d",
            "--in_channels", "3",
            "--out_channels", "5",
            "--kernel_size", "3",
            "--stride", "2",
            "--padding", "1",
            "--x", str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_no_bias(self):
        model = Conv2d(
            in_channels=4,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        x = torch.randn(1, 4, 5, 5)

        expected = model(x)

        actual = self.cli(
            "Conv2d",
            "--in_channels", "4",
            "--out_channels", "2",
            "--kernel_size", "1",
            "--stride", "1",
            "--padding", "0",
            "--bias", "false",
            "--x", str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_kernel1(self):
        model = Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        x = torch.randn(1, 8, 4, 4)

        expected = model(x)

        actual = self.cli(
            "Conv2d",
            "--in_channels", "8",
            "--out_channels", "8",
            "--kernel_size", "1",
            "--stride", "1",
            "--padding", "0",
            "--x", str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_stride_2_padding_0(self):
        model = Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=0,
        )

        x = torch.randn(1, 8, 16, 16)

        expected = model.forward(x)

        actual = self.cli(
            "Conv2d",
            "--in_channels", "8",
            "--out_channels", "8",
            "--kernel_size", "3",
            "--stride", "2",
            "--padding", "0",
            "--x", str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
