from utils import TensorTestCase
import torch

class TestTensorScalarMul(TensorTestCase):
    def test_scalar_mul_1d(self):
        b_vals = [4.0]
        pt = torch.tensor(b_vals)
        expected = 3.0 * pt
        actual = self.cli('scalar_mul', '--lhs', '3.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_scalar_mul_2d(self):
        b_vals = [[4.0], [6.0]]
        pt = torch.tensor(b_vals)
        expected = 3.0 * pt
        actual = self.cli('scalar_mul', '--lhs', '3.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
