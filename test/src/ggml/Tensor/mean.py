from utils import TensorTestCase
import torch

class TestTensorMean(TensorTestCase):
    def test_mean_1d(self):
        data = [2.0]
        pt = torch.tensor(data).float()
        expected = torch.mean(pt)
        actual = self.cli('mean', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_mean_2d(self):
        data = [[1.0], [3.0]]
        pt = torch.tensor(data).float()
        expected = torch.mean(pt)
        actual = self.cli('mean', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_mean_3d(self):
        data = list(range(1, 13))
        pt = torch.tensor(data).reshape(2, 3, 2).float()
        expected = torch.mean(pt)
        actual = self.cli('mean', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
