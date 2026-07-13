from utils import TestCase
import torch

class TestTensorSqueeze(TestCase):
    def test_squeeze(self):
        data = list(range(5))
        pt = torch.tensor(data).float().reshape(1, 5)
        expected = pt.squeeze(0)
        actual = self.cli('squeeze', '--this', str(pt.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])
    def test_squeeze_dim1(self):
        data = list(range(8))
        pt = torch.tensor(data).float().reshape(2, 1, 4)
        expected = pt.squeeze(1)
        actual = self.cli('squeeze', '--this', str(pt.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])
    def test_squeeze_last_dim(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4, 1)
        expected = pt.squeeze(2)
        actual = self.cli('squeeze', '--this', str(pt.tolist()), '--dim', '2')
        self.assertTensors(actual, [expected])
    def test_squeeze_3d(self):
        data = list(range(48))
        pt = torch.tensor(data).float().reshape(1, 6, 8)
        expected = pt.squeeze(0)
        actual = self.cli('squeeze', '--this', str(pt.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])
    def test_squeeze_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 1, 3, 4)
        expected = pt.squeeze(1)
        actual = self.cli('squeeze', '--this', str(pt.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])
