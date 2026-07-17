from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Attention
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2Attention(TestCase):
    def test_self_attention(self):
        model = Flux2Attention(
            query_dim=8,
            heads=2,
            dim_head=4,
        )

        hidden_states = torch.randn(1, 3, 8)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states)

        actual = self.cli(
            "Flux2Attention",
            "--query_dim", "8",
            "--heads", "2",
            "--dim_head", "4",

            "--hidden_states", str(hidden_states.tolist()),

            "--param-to_q-weight", str(model.to_q.weight.tolist()),
            "--param-to_k-weight", str(model.to_k.weight.tolist()),
            "--param-to_v-weight", str(model.to_v.weight.tolist()),

            "--param-norm_q-weight", str(model.norm_q.weight.tolist()),
            "--param-norm_k-weight", str(model.norm_k.weight.tolist()),

            "--param-to_out-0-weight", str(model.to_out[0].weight.tolist()),
            "--param-to_out-0-bias", str(model.to_out[0].bias.tolist()),
        )

        self.assertTensors(actual, [expected])
