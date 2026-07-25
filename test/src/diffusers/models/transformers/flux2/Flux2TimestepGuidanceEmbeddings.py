from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2TimestepGuidanceEmbeddings

class TestFlux2TimestepGuidanceEmbeddings(TestCase):
    def test(self):
        model = Flux2TimestepGuidanceEmbeddings(
            in_channels=4,
            embedding_dim=8,
        )

        timestep = torch.tensor([100.0])
        guidance = torch.tensor([3.5])

        expected = model(timestep, guidance)

        actual = self.cli(
            "Flux2TimestepGuidanceEmbeddings",
            "--in_channels", "4",
            "--embedding_dim", "8",
            "--timestep", str(timestep.tolist()),
            "--guidance", str(guidance.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])