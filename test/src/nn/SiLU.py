from utils import NNTestCase
import torch
import torch.nn as nn

class TestNNSiLU(NNTestCase):
    def test_1d(self):
        model = nn.SiLU()
        expected = self.traverse(model, torch.randn(2))
        actual = self.cli('silu', '--input', '(2)')
        self.assertEqual(actual, expected)
