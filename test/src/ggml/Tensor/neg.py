from utils import TestCase
import torch

class TestTensorNeg(TestCase):
    def test_neg_1d(self):
        data = [1.0, -2.0]
        pt = torch.tensor(data)
        expected = -pt
        actual = self.cli('neg', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_neg_2d(self):
        data = [[1.0], [-2.0]]
        pt = torch.tensor(data)
        expected = -pt
        actual = self.cli('neg', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_neg_3d(self):
        data = [[[-1.0], [2.0]], [[-3.0], [4.0]]]
        pt = torch.tensor(data)
        expected = -pt
        actual = self.cli('neg', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_neg_zeros_1d(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = -pt
        actual = self.cli('neg', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_neg_zeros_2d(self):
        data = [[0.0], [0.0]]
        pt = torch.tensor(data)
        expected = -pt
        actual = self.cli('neg', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
