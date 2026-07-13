from utils import TestCase
import torch

class TestTensorClip(TestCase):
    def test_clip_1d(self):
        data = [-5.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        actual = self.cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
    def test_clip_2d(self):
        data = [[-5.0], [-3.0]]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        actual = self.cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
    def test_clip_no_op(self):
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)
        actual = self.cli('clip', '--this', str(pt.tolist()), '--min', '-1.0', '--max', '1.0')
        self.assertTensors(actual, [expected])
