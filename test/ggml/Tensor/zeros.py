# Auto-generated from Tensor.py -- do not edit manually.
# Operator: zeros

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

class TestTensorZeros(unittest.TestCase):
    def test_zeros_1d(self):
        pt = torch.zeros(5)
        expected = pt
        result = cli('zeros', '--shape', '(5)')
        verify(self, result, expected)
    def test_zeros_2d(self):
        pt = torch.zeros(2, 3)
        expected = pt
        result = cli('zeros', '--shape', '(2, 3)')
        verify(self, result, expected)
    def test_zeros_3d(self):
        pt = torch.zeros(2, 3, 4)
        expected = pt
        result = cli('zeros', '--shape', '(2, 3, 4)')
        verify(self, result, expected)
    def test_zeros_4d(self):
        pt = torch.zeros(1, 2, 3, 4)
        expected = pt
        result = cli('zeros', '--shape', '(1, 2, 3, 4)')
        verify(self, result, expected)
