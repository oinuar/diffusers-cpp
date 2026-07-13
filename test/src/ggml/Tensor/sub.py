from utils import TestCase
import torch

class TestTensorSub(TestCase):
    def test_sub_same_shape(self):
        a = torch.randn(2, 8)
        b = torch.randn(2, 8)

        expected = a - b

        actual = self.cli(
            'sub',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_sub_trailing_dimension_broadcast(self):
        # [2, 4, 8] + [8] -> [2, 4, 8]
        a = torch.randn(2, 4, 8)
        b = torch.randn(8)

        expected = a - b

        actual = self.cli(
            'sub',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_sub_singleton_dimension_broadcast(self):
        # [2, 1, 8] + [2, 77, 8] -> [2, 77, 8]
        a = torch.randn(2, 1, 8)
        b = torch.randn(2, 77, 8)

        expected = a - b

        actual = self.cli(
            'sub',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_sub_both_tensors_expand(self):
        # [2, 1, 8] + [1, 77, 1] -> [2, 77, 8]
        a = torch.randn(2, 1, 8)
        b = torch.randn(1, 77, 1)

        expected = a - b

        actual = self.cli(
            'sub',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_sub_scalar(self):
        a = torch.randn(2, 3, 4)
        b = torch.tensor(2.0)

        expected = a - b

        actual = self.cli(
            'sub_scalar',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_scalar_sub(self):
        a = torch.tensor(2.0)
        b = torch.randn(2, 3, 4)

        expected = a - b

        actual = self.cli(
            'scalar_sub',
            '--lhs', str(a.tolist()),
            '--rhs', str(b.tolist()),
        )

        self.assertTensors(actual, [expected])
