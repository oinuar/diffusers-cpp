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
            '--param-weight', str(model.weight.tolist()),
        )

        self.assertTensors(actual, [expected], rtol=1e-3, atol=1e-5)
