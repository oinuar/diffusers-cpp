from utils import TensorTestCase
import torch

class TestTensorSubScalar(TensorTestCase):
    def test_sub_scalar_1d(self):
        a_vals = [10.0]
        pt = torch.tensor(a_vals)
        expected = pt - 5.0
        actual = self.cli('sub_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        self.assertTensors(actual, [expected])
    def test_sub_scalar_2d(self):
        a_vals = [[10.0], [20.0]]
        pt = torch.tensor(a_vals) - 5.0
        expected = pt - 5.0
        actual = self.cli('sub_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        self.assertTensors(actual, [expected])
