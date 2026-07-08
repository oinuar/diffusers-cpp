# Auto-generated from Tensor.py -- do not edit manually.
# Operator: pow

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

class TestTensorPow(unittest.TestCase):
    def test_pow_integer_1d(self):
        """[1]^2"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)
        result = cli('pow', '--this', str(pt.tolist()), '--exponent', '2.0')
        verify(self, result, expected)
    def test_pow_integer_2d(self):
        """[[1],[4]]^2"""
        data = [[1.0], [4.0]]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)
        result = cli('pow', '--this', str(pt.tolist()), '--exponent', '2.0')
        verify(self, result, expected)
    def test_pow_half(self):
        """[4]^0.5"""
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 0.5)
        result = cli('pow', '--this', str(pt.tolist()), '--exponent', '0.5')
        verify(self, result, expected)
    def test_pow_cube(self):
        """[1]^3"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 3.0)
        result = cli('pow', '--this', str(pt.tolist()), '--exponent', '3.0')
        verify(self, result, expected)
