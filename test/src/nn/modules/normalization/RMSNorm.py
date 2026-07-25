from utils import TestCase
import torch
from torch.nn.modules.normalization import RMSNorm

class TestNNNormalizationRMSNorm(TestCase):
    def test_dim3(self):
        model = RMSNorm(3)
        x = torch.randn(2, 2, 3)

        expected = model.forward(x)

        actual = self.cli(
            'RMSNorm',
            '--dim', '3',
            '--x', str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
