# Auto-generated from Tensor.py -- do not edit manually.
# Operator: abs

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

class TestTensorAbs(unittest.TestCase):
    def test_abs_1d(self):
        """abs([-3.5,-1.2])"""
        data = [-3.5]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        result = cli('abs', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_abs_2d(self):
        """abs([[-3.5,-1.2]])"""
        data = [[-3.5], [-1.2]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        result = cli('abs', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_abs_3d(self):
        """abs on rank-3 tensor"""
        data = [[[-3.5], [-1.2]], [[2.0], [-0.5]]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        result = cli('abs', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_abs_all_positive(self):
        """abs([1]) — no-op"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.abs(pt)
        result = cli('abs', '--this', str(pt.tolist()))
        verify(self, result, expected)
