from utils import NNTestCase
import torch
import torch.nn as nn

class TestNNLinear(NNTestCase):
    def test_8x4(self):
        model = nn.Linear(8, 4)
        expected = self.traverse(model, torch.randn(1, 8))
        actual = self.cli('Linear', '--in_features', '8', '--out_features', '4', '--input', '(1, 8)')
        self.assertEqual(actual, expected)
