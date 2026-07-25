from utils import TestCase
import torch
from diffusers.models.resnet import Upsample2D

class TestNNUpsample2D(TestCase):
    def test_without_conv(self):
        model = Upsample2D(
            channels=4
        )

        hidden_states = torch.randn(1, 4, 4, 4)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Upsample2D",
            "--channels", "4",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_with_conv(self):
        model = Upsample2D(
            channels=4,
            use_conv=True,
        )

        hidden_states = torch.randn(1, 4, 4, 4)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "Upsample2D",
            "--channels", "4",
            "--use_conv", "true",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
