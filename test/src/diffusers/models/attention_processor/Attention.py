from utils import TestCase
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class TestNN_Attention(TestCase):
    def test_basic(self):
        model = Attention(
            query_dim=8,
            heads=2,
            dim_head=4,
        )

        model.set_processor(AttnProcessor2_0())

        hidden_states = torch.randn(
            1,
            4,
            8,
        )

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "2",
            "--dim_head", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-to_q-weight", str(model.to_q.weight.tolist()),
            "--param-to_k-weight", str(model.to_k.weight.tolist()),
            "--param-to_v-weight", str(model.to_v.weight.tolist()),
            "--param-to_out.0-weight", str(model.to_out[0].weight.tolist()),
            "--param-to_out.0-bias", str(model.to_out[0].bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_four_heads(self):
        model = Attention(
            query_dim=16,
            heads=4,
            dim_head=4,
        )

        hidden_states = torch.randn(
            2,
            9,
            16,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "16",
            "--heads", "4",
            "--dim_head", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-to_q-weight", str(model.to_q.weight.tolist()),
            "--param-to_k-weight", str(model.to_k.weight.tolist()),
            "--param-to_v-weight", str(model.to_v.weight.tolist()),
            "--param-to_out.0-weight", str(model.to_out[0].weight.tolist()),
            "--param-to_out.0-bias", str(model.to_out[0].bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_single_token(self):
        model = Attention(
            query_dim=4,
            heads=1,
            dim_head=4,
        )

        hidden_states = torch.randn(
            1,
            1,
            4,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "4",
            "--heads", "1",
            "--dim_head", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-to_q-weight", str(model.to_q.weight.tolist()),
            "--param-to_k-weight", str(model.to_k.weight.tolist()),
            "--param-to_v-weight", str(model.to_v.weight.tolist()),
            "--param-to_out.0-weight", str(model.to_out[0].weight.tolist()),
            "--param-to_out.0-bias", str(model.to_out[0].bias.tolist()),
       )

        self.assertTensors(actual, [expected])