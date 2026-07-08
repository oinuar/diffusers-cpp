import ast
import math
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

class TestTensorPermute(unittest.TestCase):
    def test_permute_2d(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.permute(1, 0)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(1, 0)')
        verify(self, result, expected)
    def test_permute_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.permute(1, 2, 0)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(1, 2, 0)')
        verify(self, result, expected)
    def test_permute_4d(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 2, 3, 1)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(0, 2, 3, 1)')
        verify(self, result, expected)
    def test_permute_4d_inverse(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 3, 1, 2)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(0, 3, 1, 2)')
        verify(self, result, expected)
    def test_permute_negative_indices(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(-4, -2, -1, -3)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(-4, -2, -1, -3)')
        verify(self, result, expected)
