from utils import TestCase
import torch

class TestTensorSplitWithSizes(TestCase):
    def test_split_with_sizes_1d(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.split_with_sizes(pt, [2, 2])
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '2', '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_1d_uneven(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.split_with_sizes(pt, [1, 2])
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '1', '--split_size', '2')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_1d_three_parts(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.split_with_sizes(pt, [1, 2, 1])
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '1', '--split_size', '2', '--split_size', '1')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_2d_dim0(self):
        pt = torch.arange(1.0, 9.0).reshape(4, 2)
        expected = torch.split_with_sizes(pt, [3, 1], dim=0)
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '3', '--split_size', '1', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_2d_dim1(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.split_with_sizes(pt, [1, 3], dim=1)
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '1', '--split_size', '3', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_3d_dim0(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.split_with_sizes(pt, [1, 1], dim=0)
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '1', '--split_size', '1', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_split_with_sizes_3d_dim2(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 3, 4)
        expected = torch.split_with_sizes(pt, [1, 2, 1], dim=2)
        actual = self.cli('split_with_sizes', '--this', str(pt.tolist()), '--split_size', '1', '--split_size', '2', '--split_size', '1', '--dim', '2')
        self.assertTensors(actual, expected)
