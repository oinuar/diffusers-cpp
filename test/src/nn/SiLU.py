from utils import TestCase
import torch
import torch.nn as nn

class TestNNSiLU(TestCase):
    def test_1d(self):
        model = nn.SiLU()
        x = torch.randn(2)

        expected = model.forward(x)

        actual = self.cli(
            'SiLU',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])
