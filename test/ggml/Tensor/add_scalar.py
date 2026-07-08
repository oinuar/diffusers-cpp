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

class TestTensorAddScalar(unittest.TestCase):
    def test_add_scalar_1d(self):
        a_vals = [1.0]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0
        result = cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        verify(self, result, expected)
    def test_add_scalar_2d(self):
        a_vals = [[1.0], [2.0]]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0
        result = cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        verify(self, result, expected)
    def test_add_scalar_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt + 3.0
        result = cli('add_scalar', '--lhs', str(pt.tolist()), '--rhs', '3.0')
        verify(self, result, expected)
