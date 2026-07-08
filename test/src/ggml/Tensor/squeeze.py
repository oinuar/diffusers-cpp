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

class TestTensorSqueeze(unittest.TestCase):
    def test_squeeze(self):
        data = list(range(5))
        pt = torch.tensor(data).float().reshape(1, 5)
        expected = pt.squeeze(0)
        result = cli('squeeze', '--this', str(pt.tolist()), '--dim', '0')
        verify(self, result, expected)
    def test_squeeze_dim1(self):
        data = list(range(8))
        pt = torch.tensor(data).float().reshape(2, 1, 4)
        expected = pt.squeeze(1)
        result = cli('squeeze', '--this', str(pt.tolist()), '--dim', '1')
        verify(self, result, expected)
    def test_squeeze_last_dim(self):
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4, 1)
        expected = pt.squeeze(2)
        result = cli('squeeze', '--this', str(pt.tolist()), '--dim', '2')
        verify(self, result, expected)
    def test_squeeze_3d(self):
        data = list(range(48))
        pt = torch.tensor(data).float().reshape(1, 6, 8)
        expected = pt.squeeze(0)
        result = cli('squeeze', '--this', str(pt.tolist()), '--dim', '0')
        verify(self, result, expected)
    def test_squeeze_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 1, 3, 4)
        expected = pt.squeeze(1)
        result = cli('squeeze', '--this', str(pt.tolist()), '--dim', '1')
        verify(self, result, expected)
