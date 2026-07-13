from utils import TensorTestCase
import torch

class TestTensorAbs(TensorTestCase):
    def test_abs_1d(self):
        data = [-3.5]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        actual = self.cli('abs', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_abs_2d(self):
        data = [[-3.5], [-1.2]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        actual = self.cli('abs', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_abs_3d(self):
        data = [[[-3.5], [-1.2]], [[2.0], [-0.5]]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        actual = self.cli('abs', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_abs_all_positive(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        actual = self.cli('abs', '--this', str(pt.tolist()))
        self.assertTensors(actual, [expected])
