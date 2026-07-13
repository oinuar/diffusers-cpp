from utils import TensorTestCase
import torch
import math

class TestTensorSin(TensorTestCase):
    def test_sin_1d(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.sin(pt)
        actual = self.cli('sin', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sin_2d(self):
        data = [[0.0], [math.pi / 2]]
        pt = torch.tensor(data)
        expected = torch.sin(pt)
        actual = self.cli('sin', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
