# Auto-generated from Tensor.py -- do not edit manually.
# Operator: sqrt

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

class TestTensorSqrt(unittest.TestCase):
    def test_sqrt_1d(self):
        """sqrt([4])"""
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        result = cli('sqrt', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_sqrt_2d(self):
        """sqrt([[4],[9]]"""
        data = [[4.0], [9.0]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        result = cli('sqrt', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_sqrt_3d(self):
        """sqrt on rank-3 tensor"""
        data = [[[4.0], [9.0]], [[16.0], [25.0]]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        result = cli('sqrt', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_sqrt_small(self):
        """sqrt([0.25])"""
        data = [0.25]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)
        result = cli('sqrt', '--this', str(pt.tolist()))
        verify(self, result, expected)
