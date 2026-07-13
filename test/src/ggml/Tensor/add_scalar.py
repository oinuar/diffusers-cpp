from utils import TensorTestCase
import torch

class TestTensorAddScalar(TensorTestCase):
    def test_add_scalar_1d(self):
        a_vals = [1.0]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0
        actual = self.cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        self.assertTensors(actual, [expected])
    def test_add_scalar_2d(self):
        a_vals = [[1.0], [2.0]]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0
        actual = self.cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        self.assertTensors(actual, [expected])
    def test_add_scalar_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt + 3.0
        actual = self.cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        self.assertTensors(actual, [expected])
