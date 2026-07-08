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

class TestTensorArange(unittest.TestCase):
    def test_arange_default(self):
        pt = torch.arange(0.0, 5.0)
        expected = pt
        result = cli('arange', '--start', '0.0', '--stop', '5.0', '--step', '1.0')
        verify(self, result, expected)
    def test_arange_step(self):
        pt = torch.arange(0.0, 3.0, 0.5)
        expected = pt
        result = cli('arange', '--start', '0.0', '--stop', '3.0', '--step', '0.5')
        verify(self, result, expected)
    def test_arange_negative_start(self):
        pt = torch.arange(-2.0, 2.0)
        expected = pt
        result = cli('arange', '--start', '-2.0', '--stop', '2.0', '--step', '1.0')
        verify(self, result, expected)
