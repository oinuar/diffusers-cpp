from utils import TensorTestCase
import torch

class TestTensorMulScalar(TensorTestCase):
    def test_mul_scalar_1d(self):
        a_vals = [2.0]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0
        actual = self.cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        self.assertTensors(actual, [expected])
    def test_mul_scalar_2d(self):
        a_vals = [[2.0], [3.0]]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0
        actual = self.cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        self.assertTensors(actual, [expected])
    def test_mul_scalar_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt * 4.0
        actual = self.cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        self.assertTensors(actual, [expected])
