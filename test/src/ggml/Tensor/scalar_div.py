from utils import TensorTestCase
import torch

class TestTensorScalarDiv(TensorTestCase):
    def test_scalar_div_1d(self):
        b_vals = [5.0]
        pt = torch.tensor(b_vals)
        expected = 50.0 / pt
        actual = self.cli('scalar_div', '--lhs', '50.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
    def test_scalar_div_2d(self):
        b_vals = [[5.0], [10.0]]
        pt = torch.tensor(b_vals)
        expected = 50.0 / pt
        actual = self.cli('scalar_div', '--lhs', '50.0', '--rhs', str(pt.tolist()))
        self.assertTensors(actual, [expected])
