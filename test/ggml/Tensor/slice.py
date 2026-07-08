# Auto-generated from Tensor.py -- do not edit manually.
# Operator: slice

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

class TestTensorSlice(unittest.TestCase):
    def test_slice_all(self):
        """slice [:] — full slice"""
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[:]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[:]')
        verify(self, result, expected)
    def test_slice_range(self):
        """slice [1:3]"""
        data = list(range(5))
        pt = torch.tensor(data).float()
        expected = pt[1:3]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[1:3]')
        verify(self, result, expected)
    def test_slice_with_step(self):
        """slice [::2] — every other element"""
        data = list(range(5))
        pt = torch.tensor(data).float()
        expected = pt[::2]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[::2]')
        verify(self, result, expected)
    def test_slice_2d(self):
        """slice [:, 1:3] on (3,4)"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[:, 1:3]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[:, 1:3]')
        verify(self, result, expected)
    def test_slice_2d_step(self):
        """slice [::2,:] on (3,4)"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[::2, :]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[::2, :]')
        verify(self, result, expected)
    def test_slice_3d(self):
        """slice [:,:,1:] on rank-3 tensor (2,3,4)"""
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt[:, :, 1:]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[:, :, 1:]')
        verify(self, result, expected)
    def test_slice_newaxis(self):
        """slice [:, None] — insert new axis (dim=1)"""
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[:, None]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[:, None]')
        verify(self, result, expected)
    def test_slice_newaxis_2d(self):
        """slice [None,:,None] — insert new axes"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[None, :, None]
        result = cli('slice', '--this', str(pt.tolist()), '--slice', '[None, :, None]')
        verify(self, result, expected)
