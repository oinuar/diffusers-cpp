from utils import TensorTestCase
import torch

class TestTensorSlice(TensorTestCase):
    def test_slice_all(self):
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[:]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[:]')
        self.assertTensors(actual, [expected])
    def test_slice_range(self):
        data = list(range(5))
        pt = torch.tensor(data).float()
        expected = pt[1:3]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[1:3]')
        self.assertTensors(actual, [expected])
    def test_slice_with_step(self):
        data = list(range(5))
        pt = torch.tensor(data).float()
        expected = pt[::2]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[::2]')
        self.assertTensors(actual, [expected])
    def test_slice_2d(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[:, 1:3]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[:, 1:3]')
        self.assertTensors(actual, [expected])
    def test_slice_2d_step(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[::2, :]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[::2, :]')
        self.assertTensors(actual, [expected])
    def test_slice_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt[:, :, 1:]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[:, :, 1:]')
        self.assertTensors(actual, [expected])
    def test_slice_newaxis(self):
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[:, None]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[:, None]')
        self.assertTensors(actual, [expected])
    def test_slice_newaxis_2d(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[None, :, None]
        actual = self.cli('slice', '--this', str(pt.tolist()), '--slice', '[None, :, None]')
        self.assertTensors(actual, [expected])
