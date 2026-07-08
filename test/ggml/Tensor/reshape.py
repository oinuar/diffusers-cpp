# Auto-generated from Tensor.py -- do not edit manually.
# Operator: reshape

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

class TestTensorReshape(unittest.TestCase):
    def test_reshape_1d_to_2d(self):
        """reshape (6) -> (2,3)"""
        data = list(range(1, 7))
        pt = torch.tensor(data).float().reshape(6)
        expected = pt.reshape(2, 3)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(2, 3)')
        verify(self, result, expected)
    def test_reshape_2d_to_1d(self):
        """reshape (2,3) -> (6)"""
        data = list(range(1, 7))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.reshape(6)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(6)')
        verify(self, result, expected)
    def test_reshape_2d_to_4d(self):
        """reshape (6,4) -> (1,2,3,4)"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(6, 4)
        expected = pt.reshape(1, 2, 3, 4)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(1,2,3,4)')
        verify(self, result, expected)
    def test_reshape_3d_to_1d(self):
        """reshape (2,3,4) -> (24) — rank-3 tensor flattened to rank-1"""
        data = list(range(1, 25))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.reshape(24)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(24)')
        verify(self, result, expected)
    def test_reshape_3d_to_4d(self):
        """reshape (2,3,4) -> (2,2,3,2)"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.reshape(2, 2, 3, 2)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(2,2,3,2)')
        verify(self, result, expected)
    def test_reshape_4d_to_2d(self):
        """reshape (1,2,3,4) -> (6,4) — rank-4 tensor reshaped to rank-2"""
        data = list(range(1, 25))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.reshape(6, 4)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(6, 4)')
        verify(self, result, expected)
    def test_reshape_4d_to_3d(self):
        """reshape (1,2,3,4) -> (2,3,4)"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.reshape(2, 3, 4)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(2,3,4)')
        verify(self, result, expected)
    def test_reshape_infer_dimension(self):
        """reshape (6,) -> (2,-1)"""
        data = list(range(6))
        pt = torch.tensor(data).float()
        expected = pt.reshape(2, -1)
        result = cli('reshape', '--this', str(pt.tolist()), '--shape', '(2, -1)')
        verify(self, result, expected)
