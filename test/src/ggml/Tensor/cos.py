from utils import TensorTestCase
import torch
import math

class TestTensorCos(TensorTestCase):
    def test_cos_1d(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        actual = self.cli('cos', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_cos_2d(self):
        data = [[0.0], [math.pi]]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        actual = self.cli('cos', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_cos_zero(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        actual = self.cli('cos', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
