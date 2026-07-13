from utils import TestCase
import torch

class TestTensorFlatten(TestCase):
    def test_flatten_all(self):
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.flatten()
        actual = self.cli('flatten', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_flatten_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.flatten(1, 2)
        actual = self.cli('flatten', '--this', str(pt.tolist()), '--start_dim', '1', '--end_dim', '2')
        self.assertTensors(actual, [expected])
    def test_flatten_middle_dims(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.flatten(1, 2)
        actual = self.cli('flatten', '--this', str(pt.tolist()), '--start_dim', '1', '--end_dim', '2')
        self.assertTensors(actual, [expected])
    def test_flatten_all_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.flatten()
        actual = self.cli('flatten', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
