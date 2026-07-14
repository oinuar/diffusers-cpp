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
            "--param-linear_1-weight", str(model.linear_1.weight.tolist()),
            "--param-linear_1-bias", str(model.linear_1.bias.tolist()),
            "--param-linear_2-weight", str(model.linear_2.weight.tolist()),
            "--param-linear_2-bias", str(model.linear_2.bias.tolist()),
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
            "--param-linear_1-weight", str(model.linear_1.weight.tolist()),
            "--param-linear_1-bias", str(model.linear_1.bias.tolist()),
            "--param-linear_2-weight", str(model.linear_2.weight.tolist()),
            "--param-linear_2-bias", str(model.linear_2.bias.tolist()),
            "--param-cond_proj-weight", str(model.cond_proj.weight.tolist()),
        )

        self.assertTensors(actual, [expected])
