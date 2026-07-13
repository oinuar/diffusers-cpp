from utils import TensorTestCase
import torch

class TestTensorContiguous(TensorTestCase):
    def test_contiguous_1d(self):
        data = list(range(4))
        pt = torch.tensor(data).float()
        expected = pt.contiguous()
        actual = self.cli('contiguous', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_contiguous_2d(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.contiguous()
        actual = self.cli('contiguous', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_contiguous_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.contiguous()
        actual = self.cli('contiguous', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
