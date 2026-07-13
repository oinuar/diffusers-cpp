from utils import TensorTestCase
import torch

class TestTensorScalarSub(TensorTestCase):
    def test_scalar_sub_1d(self):
        b_vals = [3.0]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt
        actual = self.cli('scalar_sub', '--lhs', '20.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_scalar_sub_2d(self):
        b_vals = [[3.0], [5.0]]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt
        actual = self.cli('scalar_sub', '--lhs', '20.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
