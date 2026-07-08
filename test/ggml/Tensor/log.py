# Auto-generated from Tensor.py -- do not edit manually.
# Operator: log

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

class TestTensorLog(unittest.TestCase):
    def test_log_1d(self):
        """log([1])"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        result = cli('log', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_log_2d(self):
        """log([[1],[e]]"""
        data = [[1.0], [math.e]]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        result = cli('log', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_log_values(self):
        """log([10])"""
        data = [10.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)
        result = cli('log', '--this', str(pt.tolist()))
        verify(self, result, expected)
