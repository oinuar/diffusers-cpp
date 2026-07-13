from utils import TensorTestCase
import torch

class TestTensorMul(TensorTestCase):
    def test_mul_1d(self):
        a_vals = [2.0]
        b_vals = [4.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs
        actual = self.cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_mul_2d(self):
        a_vals = [[2.0], [3.0]]
        b_vals = [[4.0], [5.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs
        actual = self.cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_mul_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        b_vals = [float(v) + 1.0 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs * rhs
        actual = self.cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
