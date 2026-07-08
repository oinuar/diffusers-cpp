# Auto-generated from Tensor.py -- do not edit manually.
# Operator: clip

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

class TestTensorClip(unittest.TestCase):
    def test_clip_1d(self):
        """clip([-5],[-3], min=-1)"""
        data = [-5.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        result = cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        verify(self, result, expected)
    def test_clip_2d(self):
        """clip([[-5],[-3]], min=-1)"""
        data = [[-5.0], [-3.0]]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        result = cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        verify(self, result, expected)
    def test_clip_no_op(self):
        """clip([1], min=0) — no clamping"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        result = cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        verify(self, result, expected)
