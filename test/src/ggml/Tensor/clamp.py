from utils import TestCase
import torch

class TestTensorClamp(TestCase):
    def test_clamp_1d(self):
        data = [-5.0]
        pt = torch.tensor(data)
        expected = torch.clamp(pt, min=-1.0, max=1.0)
        actual = self.cli('clamp', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
    def test_clamp_2d(self):
        data = [[-5.0], [-3.0]]
        pt = torch.tensor(data)
        expected = torch.clamp(pt, min=-1.0, max=1.0)
        actual = self.cli('clamp', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
    def test_clamp_no_op(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.clamp(pt, min=-1.0, max=1.0)
        actual = self.cli('clamp', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
