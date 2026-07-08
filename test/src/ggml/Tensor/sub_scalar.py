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

class TestTensorSubScalar(unittest.TestCase):
    def test_sub_scalar_1d(self):
        a_vals = [10.0]
        pt = torch.tensor(a_vals)
        expected = pt - 5.0
        result = cli('sub_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        verify(self, result, expected)
    def test_sub_scalar_2d(self):
        a_vals = [[10.0], [20.0]]
        pt = torch.tensor(a_vals) - 5.0
        expected = pt - 5.0
        result = cli('sub_scalar', '--lhs', str(pt.tolist()), '--rhs', '5.0')
        verify(self, result, expected)
