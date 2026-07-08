# Auto-generated from Tensor.py -- do not edit manually.
# Operator: permute

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

class TestTensorPermute(unittest.TestCase):
    def test_permute_2d(self):
        """permute(1,0) on (3,4) — swap dimensions"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.permute(1, 0)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(1, 0)')
        verify(self, result, expected)
    def test_permute_3d(self):
        """permute(2,1,0) on rank-3 tensor (2,3,4) — reverse dimensions"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.permute(1, 2, 0)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(1, 2, 0)')
        verify(self, result, expected)
    def test_permute_4d(self):
        """permute(0,2,3,1) on (2,3,4,5)"""
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 2, 3, 1)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(0, 2, 3, 1)')
        verify(self, result, expected)
    def test_permute_4d_inverse(self):
        """permute(0,3,1,2) on (2,3,4,5) — the exact inverse of (0,2,3,1)"""
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(0, 3, 1, 2)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(0, 3, 1, 2)')
        verify(self, result, expected)
    def test_permute_negative_indices(self):
        """Test that negative indices resolve correctly for a 4D tensor"""
        data = list(range(120))
        pt = torch.tensor(data).float().reshape(2, 3, 4, 5)
        expected = pt.permute(-4, -2, -1, -3)
        result = cli('permute', '--this', str(pt.tolist()), '--order', '(-4, -2, -1, -3)')
        verify(self, result, expected)
