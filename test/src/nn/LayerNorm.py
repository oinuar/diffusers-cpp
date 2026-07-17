from utils import TestCase
import torch
import torch.nn as nn

class TestNNNormalizationLayerNorm(TestCase):
    def test_dim3(self):
        model = nn.modules.normalization.LayerNorm(3)
        x = torch.randn(2, 2, 3)

        expected = model.forward(x)

        actual = self.cli(
            'LayerNorm',
            '--dim', '3',
            '--x', str(x.tolist()),
            '--param-weight', str(model.weight.tolist()),
            '--param-bias', str(model.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_no_elementwise_affine(self):
        model = nn.modules.normalization.LayerNorm(3, elementwise_affine=False)
        x = torch.randn(2, 2, 3)

        expected = model.forward(x)

        actual = self.cli(
            'LayerNorm',
            '--dim', '3',
            '--elementwise_affine', 'false',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])
