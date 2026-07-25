from utils import TestCase
import torch
from diffusers.models.embeddings import TimestepEmbedding

class TestNNTimestepEmbedding(TestCase):
    def test(self):
        model = TimestepEmbedding(
            in_channels=8,
            time_embed_dim=4,
        )

        sample = torch.randn(1, 8)

        expected = model(sample)

        actual = self.cli(
            "TimestepEmbedding",
            "--in_channels", "8",
            "--time_embed_dim", "4",
            "--sample", str(sample.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_condition(self):
        model = TimestepEmbedding(
            in_channels=8,
            time_embed_dim=4,
            cond_proj_dim=2,
        )

        sample = torch.randn(1, 8)
        condition = torch.randn(1, 2)

        expected = model(sample, condition=condition)

        actual = self.cli(
            "TimestepEmbedding",
            "--in_channels", "8",
            "--time_embed_dim", "4",
            "--cond_proj_dim", "2",
            "--sample", str(sample.tolist()),
            "--condition", str(condition.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
