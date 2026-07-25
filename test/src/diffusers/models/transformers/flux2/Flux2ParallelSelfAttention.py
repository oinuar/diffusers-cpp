from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2ParallelSelfAttention
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2ParallelSelfAttention(TestCase):
    def test(self):
        model = Flux2ParallelSelfAttention(
            query_dim=8,
            heads=2,
            dim_head=4
        )

        hidden_states = torch.randn(1, 3, 8)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states)

        actual = self.cli(
            "Flux2ParallelSelfAttention",
            "--query_dim", "8",
            "--heads", "2",
            "--dim_head", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
