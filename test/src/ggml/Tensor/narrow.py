from utils import TestCase
import torch

class TestTensorNarrow(TestCase):
    def test_narrow(self):
        data = list(range(8))
        pt = torch.tensor(data).float().reshape(2, 4)
        expected = pt.narrow(1, 1, 2)
        actual = self.cli('narrow', '--this', str(pt.tolist()), '--dim', '1', '--start', '1', '--length', '2')
        self.assertTensors(actual, [expected])
    def test_narrow_3d(self):
        data = list(range(24))
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.narrow(0, 1, 1)
        actual = self.cli('narrow', '--this', str(pt.tolist()), '--dim', '0', '--start', '1', '--length', '1')
        self.assertTensors(actual, [expected])
