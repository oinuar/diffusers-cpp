from utils import TensorTestCase
import torch

class TestTensorChunk(TensorTestCase):
    def test_chunk_1d(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.chunk(pt, 2)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2')
        self.assertTensors(actual, expected)

    def test_chunk_1d_uneven(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.chunk(pt, 2)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2')
        self.assertTensors(actual, expected)

    def test_chunk_1d_three(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.chunk(pt, 4)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '4')
        self.assertTensors(actual, expected)

    def test_chunk_1d_equals_chunks(self):
        pt = torch.tensor([1.0, 2.0])
        expected = torch.chunk(pt, 1)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '1')
        self.assertTensors(actual, expected)

    def test_chunk_2d_dim0(self):
        pt = torch.arange(1.0, 9.0).reshape(4, 2)
        expected = torch.chunk(pt, 2, dim=0)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_chunk_2d_dim1(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.chunk(pt, 2, dim=1)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_chunk_3d_dim0(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.chunk(pt, 2, dim=0)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '0')
        self.assertTensors(actual, expected)

    def test_chunk_3d_dim1(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.chunk(pt, 2, dim=1)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_chunk_3d_dim2(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 3, 4)
        expected = torch.chunk(pt, 2, dim=2)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '2')
        self.assertTensors(actual, expected)

    def test_chunk_uneven_dim1(self):
        pt = torch.arange(1.0, 7.0).reshape(2, 3)
        expected = torch.chunk(pt, 2, dim=1)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2', '--dim', '1')
        self.assertTensors(actual, expected)

    def test_chunk_default_dim(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.chunk(pt, 2)
        actual = self.cli('chunk', '--this', str(pt.tolist()), '--n', '2')
        self.assertTensors(actual, expected)
