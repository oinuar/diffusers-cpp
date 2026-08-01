from utils import TestCase
import torch
from torch import nn

class TestNNEmbedding(TestCase):
    def test_1d(self):
        model = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
        )

        input = torch.tensor([0, 3, 7])

        expected = model(input)

        actual = self.cli(
            "Embedding",
            "--num_embeddings", "8",
            "--embedding_dim", "4",
            "--input", str(input.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_2d(self):
        model = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
        )

        input = torch.tensor([
            [1, 2],
            [5, 6],
        ])

        expected = model(input)

        actual = self.cli(
            "Embedding",
            "--num_embeddings", "8",
            "--embedding_dim", "4",
            "--input", str(input.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_3d(self):
        model = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
        )

        input = torch.tensor([
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ],
        ])

        expected = model(input)

        actual = self.cli(
            "Embedding",
            "--num_embeddings", "8",
            "--embedding_dim", "4",
            "--input", str(input.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_padding_idx(self):
        model = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
            padding_idx=0,
        )

        input = torch.tensor([
            [0, 1],
            [2, 0],
        ])

        expected = model(input)

        actual = self.cli(
            "Embedding",
            "--num_embeddings", "8",
            "--embedding_dim", "4",
            "--padding_idx", "0",
            "--input", str(input.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_padding_idx_negative(self):
        model = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
            padding_idx=-1,
        )

        input = torch.tensor([
            [7, 1],
            [2, 7],
        ])

        expected = model(input)

        actual = self.cli(
            "Embedding",
            "--num_embeddings", "8",
            "--embedding_dim", "4",
            "--padding_idx", "-1",
            "--input", str(input.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
