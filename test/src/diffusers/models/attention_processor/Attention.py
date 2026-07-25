from utils import TestCase
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class TestNN_Attention(TestCase):
    def test_basic(self):
        model = Attention(
            query_dim=8,
            heads=1,
            dim_head=8
        )

        hidden_states = torch.randn(
            2,
            5,
            8,
        )

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "1",
            "--dim_head", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_multihead(self):
        model = Attention(
            query_dim=16,
            heads=4,
            dim_head=4,
        )

        hidden_states = torch.randn(
            1,
            11,
            16,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "16",
            "--heads", "4",
            "--dim_head", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
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
            *self.params(model),
       )

        self.assertTensors(actual, [expected])

    def test_spatial_input(self):
        model = Attention(
            query_dim=8,
            heads=1,
            dim_head=8,
        )

        hidden_states = torch.randn(
            1,
            8,
            2,
            2,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "1",
            "--dim_head", "8",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_residual(self):
        model = Attention(
            query_dim=8,
            heads=1,
            dim_head=8,
            residual_connection=True
        )

        hidden_states = torch.randn(
            1,
            8,
            2,
            2,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "1",
            "--dim_head", "8",
            "--residual_connection", "true",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_bias(self):
        model = Attention(
            query_dim=8,
            heads=1,
            dim_head=8,
            bias=True,
            out_bias=True
        )

        hidden_states = torch.randn(
            1,
            8,
            2,
            2,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "1",
            "--dim_head", "8",
            "--bias", "true",
            "--out_bias", "true",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_bias_3d(self):
        model = Attention(
            query_dim=8,
            heads=1,
            dim_head=8,
            bias=True,
            out_bias=True
        )

        # [batch, sequence, channels]
        hidden_states = torch.randn(
            1,
            4,
            8,
        )

        expected = model(hidden_states)

        actual = self.cli(
            "Attention",
            "--query_dim", "8",
            "--heads", "1",
            "--dim_head", "8",
            "--bias", "true",
            "--out_bias", "true",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
