from utils import TestCase
import torch

class TestTensorTo(TestCase):
    def test_to_same_type(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = pt.to(torch.float32)
        actual = self.cli('to', '--this', str(pt.tolist()), '--type', '0')
        self.assertTensors(actual, [expected])
