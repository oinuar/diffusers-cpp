from utils import NNTestCase
import torch
import torch.nn as nn

class TestNNNormalizationLayerNorm(NNTestCase):
    def test_dim3(self):
        model = nn.modules.normalization.LayerNorm(3)
        expected = self.traverse(model, torch.randn(2, 2, 3))
        actual = self.cli('layernorm', '--dim', '3', '--input', '(2, 2, 3)')
        self.assertEqual(actual, expected)
