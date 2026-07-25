import unittest
import subprocess
import os
import ast
import torch

class TestCase(unittest.TestCase):
    def cli(self, *args: str) -> list:
        cli_bin = os.environ['CLI']
        result = subprocess.run([cli_bin, *args], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'{cli_bin} failed (rc={result.returncode}):\n{result.stderr}')

        outputs = []
        for line in result.stdout.strip().split('\n'):
            outputs.append(torch.tensor(ast.literal_eval(line), dtype=torch.float32))

        #print(result.stderr)

        return outputs

    def params(self, model):
        args = []

        for name, param in model.named_parameters():
            args.extend([
                f"--param-{name.replace('.', '-')}",
                str(param.tolist()),
            ])

        return args

    def assertTensors(self, actual: list, expected: list, *args: str):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.allclose(a, e, rtol=1e-4, atol=1e-6), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}')
