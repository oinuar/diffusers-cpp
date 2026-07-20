from utils import TestCase
import torch

class TestTensorTo(TestCase):
    def test_to_same_type(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = pt.to(torch.float32)
        actual = self.cli('to', '--this', str(pt.tolist()), '--type', '0')
        self.assertTensors(actual, [expected])

    def test_float_to_int(self):
        data = [3.14]
        pt = torch.tensor(data)
        expected = pt.to(torch.int32)
        actual = self.cli('to', '--this', str(pt.tolist()), '--type', '26')
        self.assertTensors(actual, [expected])
