# Auto-generated from Tensor.py -- do not edit manually.
# Operator: contiguous

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

class TestTensorContiguous(unittest.TestCase):
    def test_contiguous_1d(self):
        """contiguous() on already-contiguous tensor"""
        data = list(range(4))
        pt = torch.tensor(data).float()
        expected = pt.contiguous()
        result = cli('contiguous', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_contiguous_2d(self):
        """contiguous() on a permuted (non-contiguous) tensor"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.contiguous()
        result = cli('contiguous', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_contiguous_3d(self):
        """contiguous() on a permuted rank-3 tensor"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.contiguous()
        result = cli('contiguous', '--this', str(pt.tolist()))
        verify(self, result, expected)
