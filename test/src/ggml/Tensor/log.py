from utils import TestCase
import torch
import math

class TestTensorLog(TestCase):
    def test_log_1d(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        actual = self.cli('log', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_log_2d(self):
        data = [[1.0], [math.e]]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        actual = self.cli('log', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_log_values(self):
        data = [10.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        actual = self.cli('log', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
