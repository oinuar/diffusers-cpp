# Auto-generated from Tensor.py -- do not edit manually.
# Operator: mul

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

class TestTensorMul(unittest.TestCase):
    def test_mul_1d(self):
        """[2] * [4]"""
        a_vals = [2.0]
        b_vals = [4.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs
        result = cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        verify(self, result, expected)
    def test_mul_2d(self):
        """[[2],[3]] * [[4],[5]]"""
        a_vals = [[2.0], [3.0]]
        b_vals = [[4.0], [5.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs
        result = cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        verify(self, result, expected)
    def test_mul_3d(self):
        """[2] * [2] — rank-3 tensor"""
        data = list(range(1, 25))
        a_vals = [float(v) for v in data]
        b_vals = [float(v) + 1.0 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs * rhs
        result = cli('mul', '--lhs', str(lhs.tolist()), '--rhs', str(rhs.tolist()))
        verify(self, result, expected)
