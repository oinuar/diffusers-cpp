# Auto-generated from Tensor.py -- do not edit manually.
# Operator: mean

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

class TestTensorMean(unittest.TestCase):
    def test_mean_1d(self):
        """mean dim=-1 on (1,) -> ()"""
        data = [2.0]
        pt = torch.tensor(data).float()
        expected = torch.mean(pt)
        result = cli('mean', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_mean_2d(self):
        """mean dim=0 on (2,1) -> ()"""
        data = [[1.0], [3.0]]
        pt = torch.tensor(data).float()
        expected = torch.mean(pt)
        result = cli('mean', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_mean_3d(self):
        """mean dim=1 on rank-3 tensor (2,3,2) -> ()"""
        data = list(range(1, 13))
        pt = torch.tensor(data).reshape(2, 3, 2).float()
        expected = torch.mean(pt)
        result = cli('mean', '--this', str(pt.tolist()))
        verify(self, result, expected)
