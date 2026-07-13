from utils import TestCase
import torch

class TestTensorPow(TestCase):
    def test_pow_integer_1d(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)
        actual = self.cli('pow', '--this', str(pt.tolist()), '--exponent', '2.0')
        self.assertTensors(actual, [expected])
    def test_pow_integer_2d(self):
        data = [[1.0], [4.0]]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)
        actual = self.cli('pow', '--this', str(pt.tolist()), '--exponent', '2.0')
        self.assertTensors(actual, [expected])
    def test_pow_half(self):
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 0.5)
        actual = self.cli('pow', '--this', str(pt.tolist()), '--exponent', '0.5')
        self.assertTensors(actual, [expected])
    def test_pow_cube(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 3.0)
        actual = self.cli('pow', '--this', str(pt.tolist()), '--exponent', '3.0')
        self.assertTensors(actual, [expected])
