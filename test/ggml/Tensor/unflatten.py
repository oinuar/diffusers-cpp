# Auto-generated from Tensor.py -- do not edit manually.
# Operator: unflatten

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

class TestTensorUnflatten(unittest.TestCase):
    def test_unflatten(self):
        """unflatten dim=0 shape=(2,3) on (6,)"""
        data = [float(i) for i in range(6)]
        pt = torch.tensor(data).reshape(6)
        expected = pt.unflatten(0, (2, 3))
        result = cli('unflatten', '--this', str(pt.tolist()), '--dim', '0', '--shape', '(2, 3)')
        verify(self, result, expected)
    def test_unflatten_3d(self):
        """unflatten dim=0 shape=(3,4) on (12,)"""
        data = [float(i) for i in range(12)]
        pt = torch.tensor(data).reshape(12)
        expected = pt.unflatten(0, (3, 4))
        result = cli('unflatten', '--this', str(pt.tolist()), '--dim', '0', '--shape', '(3, 4)')
        verify(self, result, expected)
    def test_unflatten_middle_dim(self):
        """unflatten second dimension"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 12)
        expected = pt.unflatten(1, (3, 4))
        result = cli('unflatten', '--this', str(pt.tolist()), '--dim', '1', '--shape', '(3,4)')
        verify(self, result, expected)
    def test_unflatten_3d_input(self):
        """unflatten last dimension"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.unflatten(2, (2, 2))
        result = cli('unflatten', '--this', str(pt.tolist()), '--dim', '2', '--shape', '(2,2)')
        verify(self, result, expected)
