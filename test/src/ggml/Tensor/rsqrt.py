from utils import TestCase
import torch

class TestTensorRsqrt(TestCase):
    def test_rsqrt_1d(self):
        data = 4.0
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)
        actual = self.cli('rsqrt', '--this', '[' + str(pt.tolist()) + ']')
        self.assertTensors(actual, [expected])
    def test_rsqrt_2d(self):
        data = [[4.0], [25.0]]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)
        actual = self.cli('rsqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_rsqrt_3d(self):
        data = [[[4.0], [25.0]], [[9.0], [16.0]]]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)
        actual = self.cli('rsqrt', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_rsqrt_single(self):
        data = 9.0
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)
        actual = self.cli('rsqrt', '--this', '[' + str(pt.tolist()) + ']')
        self.assertTensors(actual, [expected])
