from utils import TestCase
import torch

class TestTensorMean(TestCase):
    def test_mean_dim0(self):
        x = torch.randn(2, 3, 4)

        expected = x.mean(dim=0)

        actual = self.cli(
            "mean",
            "--this", str(x.tolist()),
            "--dim", "0",
        )

        self.assertTensors(actual, [expected])

    def test_mean_dim1_keepdim(self):
        x = torch.randn(2, 3, 4)

        expected = x.mean(dim=1, keepdim=True)

        actual = self.cli(
            "mean",
            "--this", str(x.tolist()),
            "--dim", "1",
            "--keepdim", "true",
        )

        self.assertTensors(actual, [expected])

    def test_mean_negative_dim(self):
        x = torch.randn(2, 3, 4)

        expected = x.mean(dim=-1)

        actual = self.cli(
            "mean",
            "--this", str(x.tolist()),
            "--dim", "-1",
        )

        self.assertTensors(actual, [expected])
