from utils import TensorTestCase
import torch

class TestTensorScalarAdd(TensorTestCase):
    def test_scalar_add_1d(self):
        b_vals = [1.0]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt
        actual = self.cli('scalar_add', '--lhs', '10.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_scalar_add_2d(self):
        b_vals = [[1.0], [2.0]]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt
        actual = self.cli('scalar_add', '--lhs', '10.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
