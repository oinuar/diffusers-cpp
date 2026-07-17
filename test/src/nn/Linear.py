from utils import TestCase
import torch
import torch.nn as nn

class TestNNLinear(TestCase):
    def test_2d(self):
        model = nn.Linear(8, 4)
        x = torch.randn(1, 8)

        expected = model.forward(x)

        actual = self.cli(
            'Linear',
            '--in_features', '8',
            '--out_features', '4',
            '--x', str(x.tolist()),
            '--param-weight', str(model.weight.tolist()),
            '--param-bias', str(model.bias.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_3d(self):
        model = nn.Linear(8, 4)
        x = torch.randn(1, 3, 8)

        expected = model.forward(x)

        actual = self.cli(
            'Linear',
            '--in_features', '8',
            '--out_features', '4',
            '--x', str(x.tolist()),
            '--param-weight', str(model.weight.tolist()),
            '--param-bias', str(model.bias.tolist()),
        )

        self.assertTensors(actual, [expected])
