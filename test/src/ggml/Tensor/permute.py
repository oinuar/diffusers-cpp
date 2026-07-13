from utils import TestCase
import torch

class TestTensorPermute(TestCase):
    def test_permute_2d(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.permute(1, 0)
        actual = self.cli('permute', '--this', str(pt.tolist()), '--order', '(1, 0)')
        self.assertTensors(actual, [expected])
    def test_permute_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.permute(1, 2, 0)
        actual = self.cli('permute', '--this', str(pt.tolist()), '--order', '(1, 2, 0)')
        self.assertTensors(actual, [expected])
    def test_permute_4d(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 2, 3, 1)
        actual = self.cli('permute', '--this', str(pt.tolist()), '--order', '(0, 2, 3, 1)')
        self.assertTensors(actual, [expected])
    def test_permute_4d_inverse(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 3, 1, 2)
        actual = self.cli('permute', '--this', str(pt.tolist()), '--order', '(0, 3, 1, 2)')
        self.assertTensors(actual, [expected])
    def test_permute_negative_indices(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(-4, -2, -1, -3)
        actual = self.cli('permute', '--this', str(pt.tolist()), '--order', '(-4, -2, -1, -3)')
        self.assertTensors(actual, [expected])
