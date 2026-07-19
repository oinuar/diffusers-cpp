from utils import TestCase
import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

class TestNNFlux2PosEmbed(TestCase):
    def test_zero(self):
        axes_dim = [2, 2, 2, 2]

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        x = torch.randn(
            1,                      # batch
            position_ids.shape[0],  # sequence length
            4,                      # heads
            sum(axes_dim),          # head_dim
        )

        cos, sin = pos_embed(position_ids)
        expected = apply_rotary_emb(
            x,
            (cos, sin),
            sequence_dim=1,
        )

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",

            "--x", str(x.tolist()),
            "--position_ids", str(position_ids.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_4d_axes(self):
        axes_dim = [2, 2, 2, 2]

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        x = torch.randn(
            1,                      # batch
            position_ids.shape[0],  # sequence length
            4,                      # heads
            sum(axes_dim),          # head_dim
        )

        cos, sin = pos_embed(position_ids)
        expected = apply_rotary_emb(
            x,
            (cos, sin),
            sequence_dim=1,
        )

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",

            "--x", str(x.tolist()),
            "--position_ids", str(position_ids.tolist()),
        )

        self.assertTensors(actual, [expected])