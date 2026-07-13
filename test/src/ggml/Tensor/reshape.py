from utils import TensorTestCase
import torch

class TestTensorReshape(TensorTestCase):
    def test_reshape_1d_to_2d(self):
        data = list(range(1, 7))
        pt = torch.tensor(data).float().reshape(6)
        expected = pt.reshape(2, 3)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(2, 3)')
        self.assertTensors(actual, [expected])
    def test_reshape_2d_to_1d(self):
        data = list(range(1, 7))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.reshape(6)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(6)')
        self.assertTensors(actual, [expected])
    def test_reshape_2d_to_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(6, 4)
        expected = pt.reshape(1, 2, 3, 4)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(1,2,3,4)')
        self.assertTensors(actual, [expected])
    def test_reshape_3d_to_1d(self):
        data = list(range(1, 25))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.reshape(24)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(24)')
        self.assertTensors(actual, [expected])
    def test_reshape_3d_to_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.reshape(2, 2, 3, 2)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(2,2,3,2)')
        self.assertTensors(actual, [expected])
    def test_reshape_4d_to_2d(self):
        data = list(range(1, 25))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.reshape(6, 4)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(6, 4)')
        self.assertTensors(actual, [expected])
    def test_reshape_4d_to_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.reshape(2, 3, 4)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(2,3,4)')
        self.assertTensors(actual, [expected])
    def test_reshape_infer_dimension(self):
        data = list(range(6))
        pt = torch.tensor(data).float()
        expected = pt.reshape(2, -1)
        actual = self.cli('reshape', '--this', str(pt.tolist()), '--shape', '(2, -1)')
        self.assertTensors(actual, [expected])
