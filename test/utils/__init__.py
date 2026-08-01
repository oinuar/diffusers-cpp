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
            value = ast.literal_eval(line)

            # TODO: add dtype to CLI output to make this properly
            try:
                outputs.append(torch.tensor(value, dtype=torch.float32))
                continue
            except ValueError:
                pass

            outputs.append(value)

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

    def assertTensors(self, actual: list, expected: list, *args: str, **kwargs):
        finalKwargs = { 'rtol': 1e-4, 'atol': 1e-6, **kwargs }

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.allclose(a, e, **finalKwargs), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}')
