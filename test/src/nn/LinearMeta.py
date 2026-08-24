from utils import TestCase
import torch
import torch.nn as nn

class TestNNLinearMeta(TestCase):
    # out_features must be divisible by the number of meta devices (2).

    def test_2d(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8)

        expected = model.forward(x)

        actual = self.cli(
            'Linear',
            '--n-devices', '2',
            '--in_features', '8',
            '--out_features', '4',
            '--x', str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_3d(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 3, 8)

        expected = model.forward(x)

        actual = self.cli(
            'Linear',
            '--n-devices', '2',
            '--in_features', '8',
            '--out_features', '4',
            '--x', str(x.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
