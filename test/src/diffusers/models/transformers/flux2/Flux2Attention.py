from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Attention, Flux2PosEmbed
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2Attention(TestCase):
    def test(self):
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

    def test_image_rotary_emb(self):
        model = Flux2Attention(
            query_dim=16,
            heads=2,
            dim_head=8,
        )

        head_dim = model.inner_dim // model.heads
        axes_dim = [head_dim // 4] * 4

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        hidden_states = torch.randn(1, 3, 16)
        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        image_rotary_emb = pos_embed(position_ids)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states, image_rotary_emb=image_rotary_emb)

        actual = self.cli(
            "Flux2Attention",
            "--query_dim", "16",
            "--heads", "2",
            "--dim_head", "8",

            "--hidden_states", str(hidden_states.tolist()),
            "--image_rotary_emb-theta", "1000",
            "--image_rotary_emb-axes_dim", str(axes_dim[0]),
            "--image_rotary_emb-axes_dim", str(axes_dim[1]),
            "--image_rotary_emb-axes_dim", str(axes_dim[2]),
            "--image_rotary_emb-axes_dim", str(axes_dim[3]),
            "--image_rotary_emb-position_ids", str(position_ids.tolist()),

            "--param-to_q-weight", str(model.to_q.weight.tolist()),
            "--param-to_k-weight", str(model.to_k.weight.tolist()),
            "--param-to_v-weight", str(model.to_v.weight.tolist()),

            "--param-norm_q-weight", str(model.norm_q.weight.tolist()),
            "--param-norm_k-weight", str(model.norm_k.weight.tolist()),

            "--param-to_out-0-weight", str(model.to_out[0].weight.tolist()),
            "--param-to_out-0-bias", str(model.to_out[0].bias.tolist()),
        )

        self.assertTensors(actual, [expected])
