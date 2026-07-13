from utils import TensorTestCase
import torch

class TestTensorAdd(TensorTestCase):
    def test_add_1d(self):
        a_vals = [1.0]
        b_vals = [4.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs + rhs
        actual = self.cli('add', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_add_2d(self):
        a_vals = [[1.0], [3.0]]
        b_vals = [[5.0], [7.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs + rhs
        actual = self.cli('add', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_add_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        b_vals = [float(v) * 0.5 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs + rhs
        actual = self.cli('add', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
