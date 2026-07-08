# Auto-generated from Tensor.py -- do not edit manually.
# Operator: cos

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

class TestTensorCos(unittest.TestCase):
    def test_cos_1d(self):
        """cos([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        result = cli('cos', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_cos_2d(self):
        """cos([[0],[pi]]"""
        data = [[0.0], [math.pi]]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        result = cli('cos', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_cos_zero(self):
        """cos([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)
        result = cli('cos', '--this', str(pt.tolist()))
        verify(self, result, expected)
