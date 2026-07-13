from utils import TensorTestCase
import torch

class TestTensorIndex(TensorTestCase):
    def test_index_single(self):
        data = list(range(4))
        pt = torch.tensor(data).float()
        expected = pt[1]
        actual = self.cli('index', '--this', str(pt.tolist()), '--index', '1')
        self.assertTensors(actual, [expected])
    def test_index_first(self):
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[0]
        actual = self.cli('index', '--this', str(pt.tolist()), '--index', '0')
        self.assertTensors(actual, [expected])
    def test_index_2d(self):
        data = list(range(8))
        pt = torch.tensor(data).float().reshape(2, 4)
        expected = pt[1]
        actual = self.cli('index', '--this', str(pt.tolist()), '--index', '1')
        self.assertTensors(actual, [expected])
