from utils import TensorTestCase
import torch

class TestTensorSqrt(TensorTestCase):
    def test_sqrt_1d(self):
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        actual = self.cli('sqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sqrt_2d(self):
        data = [[4.0], [9.0]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        actual = self.cli('sqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sqrt_3d(self):
        data = [[[4.0], [9.0]], [[16.0], [25.0]]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        actual = self.cli('sqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_sqrt_small(self):
        data = [0.25]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        actual = self.cli('sqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
