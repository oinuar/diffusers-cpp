import unittest
import subprocess
import os
import ast
import torch

class TestCase(unittest.TestCase):
    def cli(self, *args: str) -> list:
        cli_bin = os.environ['CLI']
        n_devices = os.environ.get('N_DEVICES', 2)
        use_gpu = os.environ.get('USE_GPU', 'false')

        command = [
            *args,
            '--runner-n_devices', str(n_devices),
            '--runner-use_gpu', str(use_gpu),
            '--runner-use_local_context', "true",
        ]

        #print(" ".join([cli_bin] + list(map(lambda x: x if x.startswith("--") else f'"{x}"', command))))

        result = subprocess.run([cli_bin] + command, capture_output=True, text=True, timeout=30)

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

    def params(self, model, path=None, prefix=""):
        args = []
        param_id = 0

        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            name = f"{prefix}-{name}" if prefix else name
            name = name.replace(".", "-")
            param_id += 1

            value = str(tensor.tolist())

            if path is not None:
                filename = os.path.join(path, f"{param_id}.{prefix}-params")
                with open(filename, "w") as f:
                    f.write(value)
                value = filename

            args.extend([f"--param-{name}", value])

        return args

    def assertTensors(self, actual: list, expected: list, *args: str, **kwargs):
        finalKwargs = { 'rtol': 1e-4, 'atol': 1e-6, **kwargs }
        index = 0

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.allclose(a, e, **finalKwargs), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            index += 1
