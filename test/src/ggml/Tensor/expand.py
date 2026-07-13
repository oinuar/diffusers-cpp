from utils import TestCase
import torch

class TestTensorExpand(TestCase):
    def test_expand(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = pt.expand(2)
        actual = self.cli('expand', '--this', str(pt.tolist()), '--new-shape', '(2)')
        self.assertTensors(actual, [expected])
    def test_expand_2d(self):
        data = [1.0, 2.0]
        pt = torch.tensor(data).reshape(1, 2)
        expected = pt.expand(3, 2)
        actual = self.cli('expand', '--this', str(pt.tolist()), '--new-shape', '(3, 2)')
        self.assertTensors(actual, [expected])
