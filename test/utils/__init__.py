import unittest
import subprocess
import os
import ast
import torch

class TestCase(unittest.TestCase):
    def cli(self, *args: str) -> list:
        command = [
            *args,
            '--runner-n_devices', str(self.n_devices()),
            '--runner-use_gpu', str(self.use_gpu()).lower(),
            *sum((["--runner-tensor_split", str(x)] for x in self.tensor_splits()), [])
        ]

        command_w_args = " ".join([self.cli_bin()] + list(map(lambda x: x if x.startswith("--") else f'"{x}"', command)))

        result = subprocess.run([self.cli_bin()] + command, capture_output=True, text=True, timeout=30)

        # If command failed, write a reproduction script.
        if result.returncode != 0:
            os.makedirs(os.path.dirname(self.repro_script_path()), exist_ok=True)

            with open(self.repro_script_path(), 'w') as f:
                f.write(f"""
import os, sys, subprocess

command = [{repr(self.cli_bin())}] + {repr(command)}

env = {{
    # Put your debbing env variables here
}}

result = subprocess.run(
    command,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    env=env
)

sys.exit(result.returncode)
""")
            raise RuntimeError(f'{self.cli_bin()} failed (rc={result.returncode}):\n{result.stderr}\nRun this command to reproduce the exact failure:\n    uv run {self.repro_script_path()}')

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

        if self.verbose():
            print(result.stderr)

        return outputs

    def cli_bin(self):
        return os.environ['CLI']

    def repro_script_path(self):
        return os.path.join(os.environ['REPRO_DIR'], os.environ['TEST_ID'] + ".py")

    def n_devices(self):
        return int(os.environ.get('N_DEVICES', '1'))

    def tensor_splits(self):
        splits = os.environ.get('TENSOR_SPLITS', '')
        if splits == '':
            return []
        return splits.split(',')

    def use_gpu(self):
        return os.environ.get('USE_GPU', 'false') == 'true'

    def verbose(self):
        return os.environ.get('VERBOSE', 'false') == 'true'

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

        # GPU uses less accurate versions of operators in some cases, so make thresholds looser
        if self.use_gpu:
            finalKwargs['rtol'] = 2e-3
            finalKwargs['atol'] = 1e-3

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape, f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            self.assertEqual(a.dtype, e.dtype, f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            self.assertTrue(torch.allclose(a, e, **finalKwargs), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            index += 1
