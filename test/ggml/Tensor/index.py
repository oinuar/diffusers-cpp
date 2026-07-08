# Auto-generated from Tensor.py -- do not edit manually.
# Operator: index

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

class TestTensorIndex(unittest.TestCase):
    def test_index_single(self):
        """index [1] on (4,)"""
        data = list(range(4))
        pt = torch.tensor(data).float()
        expected = pt[1]
        result = cli('index', '--this', str(pt.tolist()), '--index', '1')
        verify(self, result, expected)
    def test_index_first(self):
        """index [0] on (3,)"""
        data = list(range(3))
        pt = torch.tensor(data).float()
        expected = pt[0]
        result = cli('index', '--this', str(pt.tolist()), '--index', '0')
        verify(self, result, expected)
    def test_index_2d(self):
        """index [1] on (2,4)"""
        data = list(range(8))
        pt = torch.tensor(data).float().reshape(2, 4)
        expected = pt[1]
        result = cli('index', '--this', str(pt.tolist()), '--index', '1')
        verify(self, result, expected)
