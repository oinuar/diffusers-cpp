from utils import TestCase
import torch
from torch.nn.modules.normalization import LayerNorm

class TestNNNormalizationLayerNorm(TestCase):
    def test_3d(self):
        model = LayerNorm(3)
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
        model = LayerNorm(3, elementwise_affine=False)
        x = torch.randn(2, 2, 3)

        expected = model.forward(x)

        actual = self.cli(
            'LayerNorm',
            '--dim', '3',
            '--elementwise_affine', 'false',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])
