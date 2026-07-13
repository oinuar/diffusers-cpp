from utils import TestCase
import torch

class TestTensorZeros(TestCase):
    def test_zeros_1d(self):
        pt = torch.zeros(5)
        expected = pt
        actual = self.cli('zeros', '--shape', '(5)')
        self.assertTensors(actual, [expected])
    def test_zeros_2d(self):
        pt = torch.zeros(2, 3)
        expected = pt
        actual = self.cli('zeros', '--shape', '(2, 3)')
        self.assertTensors(actual, [expected])
    def test_zeros_3d(self):
        pt = torch.zeros(2, 3, 4)
        expected = pt
        actual = self.cli('zeros', '--shape', '(2, 3, 4)')
        self.assertTensors(actual, [expected])
    def test_zeros_4d(self):
        pt = torch.zeros(1, 2, 3, 4)
        expected = pt
        actual = self.cli('zeros', '--shape', '(1, 2, 3, 4)')
        self.assertTensors(actual, [expected])
