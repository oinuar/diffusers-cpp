# Auto-generated from Tensor.py -- do not edit manually.
# Operator: ones

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

class TestTensorOnes(unittest.TestCase):
    def test_ones_1d(self):
        pt = torch.ones(4)
        expected = pt
        result = cli('ones', '--shape', '(4)')
        verify(self, result, expected)
    def test_ones_2d(self):
        pt = torch.ones(3, 4)
        expected = pt
        result = cli('ones', '--shape', '(3, 4)')
        verify(self, result, expected)
    def test_ones_3d(self):
        pt = torch.ones(2, 2, 3)
        expected = pt
        result = cli('ones', '--shape', '(2, 2, 3)')
        verify(self, result, expected)
    def test_ones_4d(self):
        pt = torch.ones(1, 2, 3, 4)
        expected = pt
        result = cli('ones', '--shape', '(1, 2, 3, 4)')
        verify(self, result, expected)
