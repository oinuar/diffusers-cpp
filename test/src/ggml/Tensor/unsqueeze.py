from utils import TestCase
import torch

class TestTensorUnsqueeze(TestCase):
    def test_unsqueeze(self):
        data = list(range(3))
        pt = torch.tensor(data).float().reshape(3)
        expected = pt.unsqueeze(0)
        actual = self.cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])
    def test_unsqueeze_2d(self):
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.unsqueeze(1)
        actual = self.cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])
    def test_unsqueeze_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.unsqueeze(2)
        actual = self.cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '2')
        self.assertTensors(actual, [expected])
    def test_unsqueeze_last_dim(self):
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.unsqueeze(2)
        actual = self.cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '2')
        self.assertTensors(actual, [expected])
