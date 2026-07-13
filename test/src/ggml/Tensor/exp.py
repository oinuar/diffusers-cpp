from utils import TensorTestCase
import torch

class TestTensorExp(TensorTestCase):
    def test_exp_1d(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.exp(pt)
        actual = self.cli('exp', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_exp_2d(self):
        data = [[0.0], [1.0]]
        pt = torch.tensor(data)
        expected = torch.exp(pt)
        actual = self.cli('exp', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_exp_negative(self):
        data = [-1.0]
        pt = torch.tensor(data)
        expected = torch.exp(pt)
        actual = self.cli('exp', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
