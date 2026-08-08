from utils import TestCase
import torch

class TestTensorRepeat(TestCase):
    def test(self):
        x = torch.randn(2, 3)

        expected = x.repeat(3, 2)

        actual = self.cli(
            'repeat',
            '--this', str(x.tolist()),
            '--repeats', '(3, 2)',
        )

        self.assertTensors(actual, [expected])

    def test_repeat_3d(self):
        x = torch.randn(1, 2, 3)

        expected = x.repeat(2, 3, 4)

        actual = self.cli(
            'repeat',
            '--this', str(x.tolist()),
            '--repeats', '(2, 3, 4)',
        )

        self.assertTensors(actual, [expected])
