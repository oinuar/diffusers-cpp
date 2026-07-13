from utils import TensorTestCase
import torch

class TestTensorSub(TensorTestCase):
    def test_sub_1d(self):
        a_vals = [10.0]
        b_vals = [1.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs - rhs
        actual = self.cli('sub', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_sub_2d(self):
        a_vals = [[5.0], [7.0]]
        b_vals = [[1.0], [3.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs - rhs
        actual = self.cli('sub', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_sub_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        b_vals = [float(v) * 0.5 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs - rhs
        actual = self.cli('sub', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
