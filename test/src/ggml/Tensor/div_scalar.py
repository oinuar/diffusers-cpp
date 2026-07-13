from utils import TensorTestCase
import torch

class TestTensorDivScalar(TensorTestCase):
    def test_div_scalar_1d(self):
        a_vals = [10.0]
        pt = torch.tensor(a_vals)
        expected = pt / 5.0
        actual = self.cli('div_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        self.assertTensors(actual, [expected])
    def test_div_scalar_2d(self):
        a_vals = [[10.0], [20.0]]
        pt = torch.tensor(a_vals)
        expected = pt / 5.0
        actual = self.cli('div_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        self.assertTensors(actual, [expected])
