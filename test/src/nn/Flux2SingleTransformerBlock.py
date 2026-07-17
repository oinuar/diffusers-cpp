from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2SingleTransformerBlock(TestCase):
    def test(self):
        model = Flux2SingleTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
        )

        hidden_states = torch.randn(1, 4, 8)
        temb_mod = torch.randn(1, 1, 24)  # 3 * dim

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=temb_mod,
        )

        actual = self.cli(
            "Flux2SingleTransformerBlock",

            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",

            "--hidden_states", str(hidden_states.tolist()),
            "--temb_mod", str(temb_mod.tolist()),

            "--param-attn-to_qkv_mlp_proj-weight", str(model.attn.to_qkv_mlp_proj.weight.tolist()),
            "--param-attn-norm_q-weight", str(model.attn.norm_q.weight.tolist()),
            "--param-attn-norm_k-weight", str(model.attn.norm_k.weight.tolist()),
            "--param-attn-to_out-weight", str(model.attn.to_out.weight.tolist()),
        )

        self.assertTensors(actual, [expected])