# Auto-generated from Tensor.py -- do not edit manually.
# Operator: scalar_add

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

class TestTensorScalarAdd(unittest.TestCase):
    def test_scalar_add_1d(self):
        """10 + [1]"""
        b_vals = [1.0]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt
        result = cli('scalar_add', '--lhs', '10.0', '--rhs', str(pt.tolist()))
        verify(self, result, expected)
    def test_scalar_add_2d(self):
        """10 + [[1],[2]]"""
        b_vals = [[1.0], [2.0]]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt
        result = cli('scalar_add', '--lhs', '10.0', '--rhs', str(pt.tolist()))
        verify(self, result, expected)
