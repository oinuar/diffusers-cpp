from utils import TestCase
import torch
from diffusers.models.normalization import AdaLayerNormContinuous

class TestNNAdaLayerNormContinuous(TestCase):
    def test_8x4(self):
        model = AdaLayerNormContinuous(
            embedding_dim=8,
            conditioning_embedding_dim=4,
        )

        hidden_states = torch.randn(1, 8)
        conditioning_embedding = torch.randn(1, 4)

        expected = model.forward(hidden_states, conditioning_embedding)

        actual = self.cli(
            'AdaLayerNormContinuous',
            '--embedding_dim', '8',
            '--conditioning_embedding_dim', '4',
            '--hidden_states', str(hidden_states.tolist()),
            '--conditioning_embedding', str(conditioning_embedding.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
