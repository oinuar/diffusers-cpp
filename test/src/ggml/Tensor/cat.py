import ast
import subprocess
import unittest
import torch
import os

def cli(*args: str) -> torch.Tensor:
    cli = os.environ.get('TENSOR_CLI', 'tensor-cli')
    result = subprocess.run([cli, *args], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'tensor-cli failed (rc={result.returncode}):\n{result.stderr}')
    return torch.tensor(ast.literal_eval(result.stdout), dtype=torch.float32)

def verify(self, actual: torch.Tensor, expected: torch.Tensor, rtol: float=1e-05, atol: float=1e-08):
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol), f'\nActual: {str(actual.tolist())}\nExpected: {str(expected.tolist())}')

class TestTensorCat(unittest.TestCase):
    def test_cat_1d_dim0(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        expected = torch.cat([a, b], dim=0)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        verify(self, result, expected)

    def test_cat_2d_dim0(self):
        a = torch.tensor([[1.0], [2.0]])
        b = torch.tensor([[3.0], [4.0]])
        expected = torch.cat([a, b], dim=0)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        verify(self, result, expected)

    def test_cat_2d_dim1(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        expected = torch.cat([a, b], dim=1)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        verify(self, result, expected)

    def test_cat_3d_dim0(self):
        a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        b = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        expected = torch.cat([a, b], dim=0)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        verify(self, result, expected)

    def test_cat_3d_dim1(self):
        a = torch.tensor([[[1.0, 2.0]]])
        b = torch.tensor([[[3.0, 4.0]]])
        expected = torch.cat([a, b], dim=1)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        verify(self, result, expected)

    def test_cat_3d_dim2(self):
        a = torch.tensor([[[1.0], [2.0]]])
        b = torch.tensor([[[3.0], [4.0]]])
        expected = torch.cat([a, b], dim=2)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '2')
        verify(self, result, expected)

    def test_cat_three_tensors(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        expected = torch.cat([a, b, c], dim=0)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--tensor', str(c.tolist()), '--dim', '0')
        verify(self, result, expected)

    def test_cat_2d_different_sizes_dim1(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0], [6.0]])
        expected = torch.cat([a, b], dim=1)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '1')
        verify(self, result, expected)

    def test_cat_2d_different_sizes_dim0(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        expected = torch.cat([a, b], dim=0)
        result = cli('cat', '--tensor', str(a.tolist()), '--tensor', str(b.tolist()), '--dim', '0')
        verify(self, result, expected)
