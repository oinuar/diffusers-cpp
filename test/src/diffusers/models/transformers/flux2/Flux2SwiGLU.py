from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU

class TestFlux2SwiGLU(TestCase):
    def test_1d(self):
        model = Flux2SwiGLU()
        gate = torch.randn(2)
        up = torch.randn(2)

        # The C++ side computes the split form: silu(gate) * up, while the
        # reference chunks the fused (gate | up) tensor.
        expected = model.forward(torch.cat([gate, up]))

        actual = self.cli(
            'Flux2SwiGLU',
            '--gate', str(gate.tolist()),
            '--up', str(up.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_2d(self):
            model = Flux2SwiGLU()
            gate = torch.randn(2, 12)
            up = torch.randn(2, 12)

            expected = model.forward(torch.cat([gate, up], dim=-1))

            actual = self.cli(
                'Flux2SwiGLU',
                '--gate', str(gate.tolist()),
                '--up', str(up.tolist()),
            )

            self.assertTensors(actual, [expected])
