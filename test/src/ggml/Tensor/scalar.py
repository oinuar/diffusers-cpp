from utils import TestCase
import torch

class TestTensorScalar(TestCase):
    def test_scalar(self):
        pt = torch.tensor(42.0)
        expected = pt
        actual = self.cli('scalar', '--value', '42.0')
        self.assertTensors(actual, [expected])
    def test_scalar_negative(self):
        pt = torch.tensor(-3.14)
        expected = pt
        actual = self.cli('scalar', '--value', '-3.14')
        self.assertTensors(actual, [expected])
