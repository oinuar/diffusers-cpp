# Auto-generated from Tensor.py -- do not edit manually.
# Operator: unsqueeze

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

class TestTensorUnsqueeze(unittest.TestCase):
    def test_unsqueeze(self):
        """unsqueeze dim=0 on (N,) -> (1,N)"""
        data = list(range(3))
        pt = torch.tensor(data).float().reshape(3)
        expected = pt.unsqueeze(0)
        result = cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '0')
        verify(self, result, expected)
    def test_unsqueeze_2d(self):
        """unsqueeze dim=1 on (N,M) -> (N,1,M)"""
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.unsqueeze(1)
        result = cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '1')
        verify(self, result, expected)
    def test_unsqueeze_4d(self):
        """unsqueeze into rank-4"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.unsqueeze(2)
        result = cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '2')
        verify(self, result, expected)
    def test_unsqueeze_last_dim(self):
        """unsqueeze at end"""
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.unsqueeze(2)
        result = cli('unsqueeze', '--this', str(pt.tolist()), '--dim', '2')
        verify(self, result, expected)
