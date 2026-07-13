from utils import TestCase
import torch

class TestTensorSum(TestCase):
    def test_sum_1d(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.sum(pt)
        actual = self.cli('sum', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sum_2d(self):
        data = [[1.0], [2.0]]
        pt = torch.tensor(data)
        expected = torch.sum(pt)
        actual = self.cli('sum', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sum_3d(self):
        data = [[[-1.0], [2.0], [3.0]], [[4.0], [-5.0], [6.0]]]
        pt = torch.tensor(data)
        expected = torch.sum(pt)
        actual = self.cli('sum', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sum_negative(self):
        data = [-1.0]
        pt = torch.tensor(data)
        expected = torch.sum(pt)
        actual = self.cli('sum', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
