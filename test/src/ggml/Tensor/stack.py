from utils import TestCase
import torch

class TestTensorStack(TestCase):
    def test(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)
        c = torch.randn(2, 3, 4)

        expected = torch.stack([a, b, c], dim=1)

        actual = self.cli(
            'stack',
            '--tensor', str(a.tolist()),
            '--tensor', str(b.tolist()),
            '--tensor', str(c.tolist()),
            '--dim', '1',
        )

        self.assertTensors(actual, [expected])

    def test_negative_dim(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)

        expected = torch.stack([a, b], dim=-1)

        actual = self.cli(
            'stack',
            '--tensor', str(a.tolist()),
            '--tensor', str(b.tolist()),
            '--dim', '-1',
        )

        self.assertTensors(actual, [expected])
