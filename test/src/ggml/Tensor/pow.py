from utils import TestCase
import torch

class TestTensorPow(TestCase):
    def test_scalar(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)
        actual = self.cli('pow_scalar', '--base', str(pt.tolist()), '--exponent', '2.0')
        self.assertTensors(actual, [expected])

