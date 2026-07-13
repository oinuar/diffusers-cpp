from utils import TensorTestCase
import torch

class TestTensorDiv(TensorTestCase):
    def test_div_1d(self):
        a_vals = [10.0]
        b_vals = [2.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs / rhs
        actual = self.cli('div', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_div_2d(self):
        a_vals = [[10.0], [30.0]]
        b_vals = [[2.5], [6.5]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs / rhs
        actual = self.cli('div', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
    def test_div_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) + 1.0 for v in data]
        b_vals = [float(v) + 2.0 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs / rhs
        actual = self.cli('div', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        self.assertTensors(actual, [expected])
