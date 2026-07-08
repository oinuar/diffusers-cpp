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

class TestTensorFlatten(unittest.TestCase):
    def test_flatten_all(self):
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.flatten()
        result = cli('flatten', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_flatten_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.flatten(1, 2)
        result = cli('flatten', '--this', str(pt.tolist()), '--start_dim', '1', '--end_dim', '2')
        verify(self, result, expected)
    def test_flatten_middle_dims(self):
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.flatten(1, 2)
        result = cli('flatten', '--this', str(pt.tolist()), '--start_dim', '1', '--end_dim', '2')
        verify(self, result, expected)
    def test_flatten_all_4d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.flatten()
        result = cli('flatten', '--this', str(pt.tolist()))
        verify(self, result, expected)
