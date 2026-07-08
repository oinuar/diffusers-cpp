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

class TestTensorNeg(unittest.TestCase):
    def test_neg_1d(self):
        data = [1.0, -2.0]
        pt = torch.tensor(data)
        expected = -pt
        result = cli('neg', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_neg_2d(self):
        data = [[1.0], [-2.0]]
        pt = torch.tensor(data)
        expected = -pt
        result = cli('neg', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_neg_3d(self):
        data = [[[-1.0], [2.0]], [[-3.0], [4.0]]]
        pt = torch.tensor(data)
        expected = -pt
        result = cli('neg', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_neg_zeros_1d(self):
        data = [0.0]
        pt = torch.tensor(data)
        expected = -pt
        result = cli('neg', '--this', str(pt.tolist()))
        verify(self, result, expected)
    def test_neg_zeros_2d(self):
        data = [[0.0], [0.0]]
        pt = torch.tensor(data)
        expected = -pt
        result = cli('neg', '--this', str(pt.tolist()))
        verify(self, result, expected)
