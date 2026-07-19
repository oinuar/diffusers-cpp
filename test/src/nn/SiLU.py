from utils import TestCase
import torch
from torch.nn import SiLU

class TestNNSiLU(TestCase):
    def test_1d(self):
        model = SiLU()
        x = torch.randn(2)

        expected = model.forward(x)

        actual = self.cli(
            'SiLU',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])
