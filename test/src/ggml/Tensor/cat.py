from utils import TensorTestCase
import torch

class TestTensorCat(TensorTestCase):
    def test_cat_1d_dim0(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        expected = torch.cat([a, b], dim=0)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])

    def test_cat_2d_dim0(self):
        a = torch.tensor([[1.0], [2.0]])
        b = torch.tensor([[3.0], [4.0]])
        expected = torch.cat([a, b], dim=0)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])

    def test_cat_2d_dim1(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        expected = torch.cat([a, b], dim=1)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])

    def test_cat_3d_dim0(self):
        a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        b = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        expected = torch.cat([a, b], dim=0)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])

    def test_cat_3d_dim1(self):
        a = torch.tensor([[[1.0, 2.0]]])
        b = torch.tensor([[[3.0, 4.0]]])
        expected = torch.cat([a, b], dim=1)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])

    def test_cat_3d_dim2(self):
        a = torch.tensor([[[1.0], [2.0]]])
        b = torch.tensor([[[3.0], [4.0]]])
        expected = torch.cat([a, b], dim=2)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '2')
        self.assertTensors(actual, [expected])

    def test_cat_three_tensors(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        expected = torch.cat([a, b, c], dim=0)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--tensor', str(c.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])

    def test_cat_2d_different_sizes_dim1(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0], [6.0]])
        expected = torch.cat([a, b], dim=1)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        self.assertTensors(actual, [expected])

    def test_cat_2d_different_sizes_dim0(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        expected = torch.cat([a, b], dim=0)
        actual = self.cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        self.assertTensors(actual, [expected])
