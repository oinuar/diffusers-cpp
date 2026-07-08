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

class TestTensorMulScalar(unittest.TestCase):
    def test_mul_scalar_1d(self):
        a_vals = [2.0]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0
        result = cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        verify(self, result, expected)
    def test_mul_scalar_2d(self):
        a_vals = [[2.0], [3.0]]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0
        result = cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        verify(self, result, expected)
    def test_mul_scalar_3d(self):
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt * 4.0
        result = cli('mul_scalar', '--lhs', str(pt.tolist()), '--rhs', '4.0')
        verify(self, result, expected)
