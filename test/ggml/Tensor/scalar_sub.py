# Auto-generated from Tensor.py -- do not edit manually.
# Operator: scalar_sub

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

class TestTensorScalarSub(unittest.TestCase):
    def test_scalar_sub_1d(self):
        """20 - [3]"""
        b_vals = [3.0]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt
        result = cli('scalar_sub', '--lhs', '20.0', '--rhs', str(pt.tolist()))
        verify(self, result, expected)
    def test_scalar_sub_2d(self):
        """20 - [[3],[5]]"""
        b_vals = [[3.0], [5.0]]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt
        result = cli('scalar_sub', '--lhs', '20.0', '--rhs', str(pt.tolist()))
        verify(self, result, expected)
