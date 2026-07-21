from utils import TestCase
import torch

class TestTensorSum(TestCase):
    def test_sum_dim0(self):
        x = torch.randn(2, 3, 4)

        expected = x.sum(dim=0)

        actual = self.cli(
            "sum",
            "--this", str(x.tolist()),
            "--dim", "0",
        )

        self.assertTensors(actual, [expected])

    def test_sum_dim1_keepdim(self):
        x = torch.randn(2, 3, 4)

        expected = x.sum(dim=1, keepdim=True)

        actual = self.cli(
            "sum",
            "--this", str(x.tolist()),
            "--dim", "1",
            "--keepdim", "true",
        )

        self.assertTensors(actual, [expected])

    def test_sum_negative_dim(self):
        x = torch.randn(2, 3, 4)

        expected = x.sum(dim=-1)

        actual = self.cli(
            "sum",
            "--this", str(x.tolist()),
            "--dim", "-1",
        )

        self.assertTensors(actual, [expected])
