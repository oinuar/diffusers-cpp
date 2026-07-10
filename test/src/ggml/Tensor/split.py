import ast
import os
import subprocess
import unittest
import torch


def cli(*args: str) -> list:
    cli_bin = os.environ.get('TENSOR_CLI', 'tensor-cli')
    result = subprocess.run([cli_bin, *args], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'tensor-cli failed (rc={result.returncode}):\n{result.stderr}')

    outputs = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        outputs.append(torch.tensor(ast.literal_eval(line), dtype=torch.float32))

    return outputs


def verify(self, actual: list, expected: list):
    self.assertEqual(len(actual), len(expected))
    for a, e in zip(actual, expected):
        self.assertEqual(a.shape, e.shape)
        self.assertTrue(torch.allclose(a.float(), e.float()), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}')


class TestTensorSplit(unittest.TestCase):
    def test_split_1d_even(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.split(pt, 2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        verify(self, result, expected)

    def test_split_1d_even_n3(self):
        pt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = torch.split(pt, 2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        verify(self, result, expected)

    def test_split_1d_uneven(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.split(pt, 2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        verify(self, result, expected)

    def test_split_1d_size_larger(self):
        pt = torch.tensor([1.0, 2.0])
        expected = torch.split(pt, 5)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '5')
        verify(self, result, expected)

    def test_split_1d_size_equals(self):
        pt = torch.tensor([1.0, 2.0])
        expected = torch.split(pt, 2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        verify(self, result, expected)

    def test_split_1d_size_1(self):
        pt = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.split(pt, 1)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '1')
        verify(self, result, expected)

    def test_split_2d_dim0(self):
        pt = torch.arange(1.0, 9.0).reshape(4, 2)
        expected = torch.split(pt, 2, dim=0)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '0')
        verify(self, result, expected)

    def test_split_2d_dim1(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.split(pt, 2, dim=1)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        verify(self, result, expected)

    def test_split_2d_dim0_uneven(self):
        pt = torch.arange(1.0, 7.0).reshape(2, 3)
        expected = torch.split(pt, 1, dim=0)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '1', '--dim', '0')
        verify(self, result, expected)

    def test_split_2d_dim1_uneven(self):
        pt = torch.arange(1.0, 7.0).reshape(2, 3)
        expected = torch.split(pt, 2, dim=1)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        verify(self, result, expected)

    def test_split_3d_dim0(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.split(pt, 1, dim=0)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '1', '--dim', '0')
        verify(self, result, expected)

    def test_split_3d_dim1(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 4, 3)
        expected = torch.split(pt, 2, dim=1)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '1')
        verify(self, result, expected)

    def test_split_3d_dim2(self):
        pt = torch.arange(1.0, 25.0).reshape(2, 3, 4)
        expected = torch.split(pt, 2, dim=2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2', '--dim', '2')
        verify(self, result, expected)

    def test_split_default_dim(self):
        pt = torch.arange(1.0, 9.0).reshape(2, 4)
        expected = torch.split(pt, 2)
        result = cli('split', '--this', str(pt.tolist()), '--split_size', '2')
        verify(self, result, expected)
