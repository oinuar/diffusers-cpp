import unittest
import subprocess
import os
import ast
import torch

class TestCase(unittest.TestCase):
    def cli(self, *args: str) -> list:
        cli_bin = os.environ['CLI']
        print(" ".join([cli_bin] + list(map(lambda x: x if x.startswith("--") else f'"{x}"', [*args]))))
        result = subprocess.run([cli_bin, *args], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'{cli_bin} failed (rc={result.returncode}):\n{result.stderr}')

        outputs = []
        for line in result.stdout.strip().split('\n'):
            outputs.append(torch.tensor(ast.literal_eval(line), dtype=torch.float32))

        return outputs

    def assertTensors(self, actual: list, expected: list, rtol: float=1e-4, atol: float=1e-6):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.allclose(a, e, rtol=rtol, atol=atol), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}')
