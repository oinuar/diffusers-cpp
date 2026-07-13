from utils import TestCase
import torch

class TestTensorOnes(TestCase):
    def test_ones_1d(self):
        pt = torch.ones(4)
        expected = pt
        actual = self.cli('ones', '--shape', '(4)')
        self.assertTensors(actual, [expected])
    def test_ones_2d(self):
        pt = torch.ones(3, 4)
        expected = pt
        actual = self.cli('ones', '--shape', '(3, 4)')
        self.assertTensors(actual, [expected])
    def test_ones_3d(self):
        pt = torch.ones(2, 2, 3)
        expected = pt
        actual = self.cli('ones', '--shape', '(2, 2, 3)')
        self.assertTensors(actual, [expected])
    def test_ones_4d(self):
        pt = torch.ones(1, 2, 3, 4)
        expected = pt
        actual = self.cli('ones', '--shape', '(1, 2, 3, 4)')
        self.assertTensors(actual, [expected])
