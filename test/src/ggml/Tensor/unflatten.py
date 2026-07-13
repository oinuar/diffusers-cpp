from utils import TensorTestCase
import torch

class TestTensorUnflatten(TensorTestCase):
    def test_unflatten(self):
        data = [float(i) for i in range(6)]
        pt = torch.tensor(data).reshape(6)
        expected = pt.unflatten(0, (2, 3))
        actual = self.cli('unflatten', '--this', str(pt.tolist()), '--dim', '0', '--shape', '(2, 3)')
        self.assertTensors(actual, [expected])
    def test_unflatten_3d(self):
        data = [float(i) for i in range(12)]
        pt = torch.tensor(data).reshape(12)
        expected = pt.unflatten(0, (3, 4))
        actual = self.cli('unflatten', '--this', str(pt.tolist()), '--dim', '0', '--shape', '(3, 4)')
        self.assertTensors(actual, [expected])
    def test_unflatten_middle_dim(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 12)
        expected = pt.unflatten(1, (3, 4))
        actual = self.cli('unflatten', '--this', str(pt.tolist()), '--dim', '1', '--shape', '(3,4)')
        self.assertTensors(actual, [expected])
    def test_unflatten_3d_input(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.unflatten(2, (2, 2))
        actual = self.cli('unflatten', '--this', str(pt.tolist()), '--dim', '2', '--shape', '(2,2)')
        self.assertTensors(actual, [expected])
