from utils import TestCase
import torch

class TestTensorSplit(TestCase):
    def test_split_1d_even(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.split(pt, 2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_1d_even_n3(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = torch.split(pt, 2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_1d_uneven(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.split(pt, 2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_1d_size_larger(self):
        pt = torch.tensor([1.0, 2.0])
        expected = torch.split(pt, 5)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '5')
        self.assertTensors(actual, expected)

    def test_split_1d_size_equals(self):
        pt = torch.tensor([1.0, 2.0])
        expected = torch.split(pt, 2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_1d_size_1(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.split(pt, 1)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '1')
        self.assertTensors(actual, expected)

    def test_split_2d_dim0(self):
        pt = torch.arange(1.0, 9.0).reshape(4, 2)
        expected = torch.split(pt, 2, dim=0)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_split_2d_dim1(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.split(pt, 2, dim=1)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_split_2d_dim0_uneven(self):
        pt = torch.arange(1.0, 7.0).reshape(2, 3)
        expected = torch.split(pt, 1, dim=0)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '1', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_split_2d_dim1_uneven(self):
        pt = torch.arange(1.0, 7.0).reshape(2, 3)
        expected = torch.split(pt, 2, dim=1)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_split_3d_dim0(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.split(pt, 1, dim=0)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '1', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_split_3d_dim1(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.split(pt, 2, dim=1)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_split_3d_dim2(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 3, 4)
        expected = torch.split(pt, 2, dim=2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '2')
        self.assertTensors(actual, expected)

    def test_split_default_dim(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.split(pt, 2)
        actual = self.cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        self.assertTensors(actual, expected)
