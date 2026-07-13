from utils import TestCase
import torch

class TestTensorArange(TestCase):
    def test_arange_default(self):
        pt = torch.arange(0.0, 5.0)
        expected = pt
        actual = self.cli('arange', '--start', '0.0', '--stop', '5.0', '--step', '1.0')
        self.assertTensors(actual, [expected])
    def test_arange_step(self):
        pt = torch.arange(0.0, 3.0, 0.5)
        expected = pt
        actual = self.cli('arange', '--start', '0.0', '--stop', '3.0', '--step', '0.5')
        self.assertTensors(actual, [expected])
    def test_arange_negative_start(self):
        pt = torch.arange(-2.0, 2.0)
        expected = pt
        actual = self.cli('arange', '--start', '-2.0', '--stop', '2.0', '--step', '1.0')
        self.assertTensors(actual, [expected])
