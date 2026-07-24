from utils import TestCase
import torch
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D

class TestNNUpDecoderBlock2D(TestCase):
    def test_no_upsample(self):
        model = UpDecoderBlock2D(
            num_layers=2,
            in_channels=8,
            out_channels=4,
            add_upsample=False,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
            temb_channels=None,
        )

        hidden_states = torch.randn(1, 8, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "UpDecoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "8",
            "--out_channels", "4",
            "--add_upsample", "false",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-resnets-1-conv2-bias", str(model.resnets[1].conv2.bias.tolist()),
            "--param-resnets-1-conv2-weight", str(model.resnets[1].conv2.weight.tolist()),
            "--param-resnets-1-norm2-bias", str(model.resnets[1].norm2.bias.tolist()),
            "--param-resnets-1-norm2-weight", str(model.resnets[1].norm2.weight.tolist()),
            "--param-resnets-1-conv1-bias", str(model.resnets[1].conv1.bias.tolist()),
            "--param-resnets-1-conv1-weight", str(model.resnets[1].conv1.weight.tolist()),
            "--param-resnets-1-norm1-bias", str(model.resnets[1].norm1.bias.tolist()),
            "--param-resnets-1-norm1-weight", str(model.resnets[1].norm1.weight.tolist()),
            "--param-resnets-0-conv2-bias", str(model.resnets[0].conv2.bias.tolist()),
            "--param-resnets-0-conv2-weight", str(model.resnets[0].conv2.weight.tolist()),
            "--param-resnets-0-norm2-bias", str(model.resnets[0].norm2.bias.tolist()),
            "--param-resnets-0-norm2-weight", str(model.resnets[0].norm2.weight.tolist()),
            "--param-resnets-0-conv1-bias", str(model.resnets[0].conv1.bias.tolist()),
            "--param-resnets-0-conv1-weight", str(model.resnets[0].conv1.weight.tolist()),
            "--param-resnets-0-conv_shortcut-bias", str(model.resnets[0].conv_shortcut.bias.tolist()),
            "--param-resnets-0-conv_shortcut-weight", str(model.resnets[0].conv_shortcut.weight.tolist()),
            "--param-resnets-0-norm1-bias", str(model.resnets[0].norm1.bias.tolist()),
            "--param-resnets-0-norm1-weight", str(model.resnets[0].norm1.weight.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_with_upsample(self):
        model = UpDecoderBlock2D(
            num_layers=2,
            in_channels=8,
            out_channels=4,
            add_upsample=True,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=4,
            temb_channels=None,
        )

        hidden_states = torch.randn(1, 8, 8, 8)

        expected = model.forward(hidden_states)

        actual = self.cli(
            "UpDecoderBlock2D",
            "--num_layers", "2",
            "--in_channels", "8",
            "--out_channels", "4",
            "--add_upsample", "true",
            "--resnet_groups", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--param-upsamplers-0-conv-bias", str(model.upsamplers[0].conv.bias.tolist()),
            "--param-upsamplers-0-conv-weight", str(model.upsamplers[0].conv.weight.tolist()),
            "--param-resnets-1-conv2-bias", str(model.resnets[1].conv2.bias.tolist()),
            "--param-resnets-1-conv2-weight", str(model.resnets[1].conv2.weight.tolist()),
            "--param-resnets-1-norm2-bias", str(model.resnets[1].norm2.bias.tolist()),
            "--param-resnets-1-norm2-weight", str(model.resnets[1].norm2.weight.tolist()),
            "--param-resnets-1-conv1-bias", str(model.resnets[1].conv1.bias.tolist()),
            "--param-resnets-1-conv1-weight", str(model.resnets[1].conv1.weight.tolist()),
            "--param-resnets-1-norm1-bias", str(model.resnets[1].norm1.bias.tolist()),
            "--param-resnets-1-norm1-weight", str(model.resnets[1].norm1.weight.tolist()),
            "--param-resnets-0-conv2-bias", str(model.resnets[0].conv2.bias.tolist()),
            "--param-resnets-0-conv2-weight", str(model.resnets[0].conv2.weight.tolist()),
            "--param-resnets-0-norm2-bias", str(model.resnets[0].norm2.bias.tolist()),
            "--param-resnets-0-norm2-weight", str(model.resnets[0].norm2.weight.tolist()),
            "--param-resnets-0-conv1-bias", str(model.resnets[0].conv1.bias.tolist()),
            "--param-resnets-0-conv1-weight", str(model.resnets[0].conv1.weight.tolist()),
            "--param-resnets-0-conv_shortcut-bias", str(model.resnets[0].conv_shortcut.bias.tolist()),
            "--param-resnets-0-conv_shortcut-weight", str(model.resnets[0].conv_shortcut.weight.tolist()),
            "--param-resnets-0-norm1-bias", str(model.resnets[0].norm1.bias.tolist()),
            "--param-resnets-0-norm1-weight", str(model.resnets[0].norm1.weight.tolist()),
        )

        self.assertTensors(actual, [expected])
