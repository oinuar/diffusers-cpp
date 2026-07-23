from utils import TestCase
import torch
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D

class TestNN_UNetMidBlock2D(TestCase):
    def test_without_attention(self):
        model = UNetMidBlock2D(
            in_channels=8,
            attention_head_dim=8,
            resnet_groups=4,
            temb_channels=None,
            add_attention=False,
            resnet_act_fn="silu",
            resnet_time_scale_shift="default",
        )

        sample = torch.randn(
            1,
            8,
            4,
            4,
        )

        expected = model(sample)

        actual = self.cli(
            "UNetMidBlock2D",
            "--in_channels", "8",
            "--attention_head_dim", "8",
            "--resnet_groups", "4",
            "--add_attention", "false",
            "--sample", str(sample.tolist()),
            "--param-resnets.1-conv2-bias", str(model.resnets[1].conv2.bias.tolist()),
            "--param-resnets.1-conv2-weight", str(model.resnets[1].conv2.weight.tolist()),
            "--param-resnets.1-norm2-bias", str(model.resnets[1].norm2.bias.tolist()),
            "--param-resnets.1-norm2-weight", str(model.resnets[1].norm2.weight.tolist()),
            "--param-resnets.1-conv1-bias", str(model.resnets[1].conv1.bias.tolist()),
            "--param-resnets.1-conv1-weight", str(model.resnets[1].conv1.weight.tolist()),
            "--param-resnets.1-norm1-bias", str(model.resnets[1].norm1.bias.tolist()),
            "--param-resnets.1-norm1-weight", str(model.resnets[1].norm1.weight.tolist()),
            "--param-resnets.0-conv2-bias", str(model.resnets[0].conv2.bias.tolist()),
            "--param-resnets.0-conv2-weight", str(model.resnets[0].conv2.weight.tolist()),
            "--param-resnets.0-norm2-bias", str(model.resnets[0].norm2.bias.tolist()),
            "--param-resnets.0-norm2-weight", str(model.resnets[0].norm2.weight.tolist()),
            "--param-resnets.0-conv1-bias", str(model.resnets[0].conv1.bias.tolist()),
            "--param-resnets.0-conv1-weight", str(model.resnets[0].conv1.weight.tolist()),
            "--param-resnets.0-norm1-bias", str(model.resnets[0].norm1.bias.tolist()),
            "--param-resnets.0-norm1-weight", str(model.resnets[0].norm1.weight.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_with_attention(self):
        model = UNetMidBlock2D(
            in_channels=8,
            attention_head_dim=8,
            resnet_groups=4,
            temb_channels=None,
            add_attention=True,
            resnet_act_fn="silu",
            resnet_time_scale_shift="default",
        )

        sample = torch.randn(
            1,
            8,
            4,
            4,
        )

        expected = model(sample)

        actual = self.cli(
            "UNetMidBlock2D",
            "--in_channels", "8",
            "--attention_head_dim", "8",
            "--resnet_groups", "4",
            "--sample", str(sample.tolist()),
            "--param-resnets.1-conv2-bias", str(model.resnets[1].conv2.bias.tolist()),
            "--param-resnets.1-conv2-weight", str(model.resnets[1].conv2.weight.tolist()),
            "--param-resnets.1-norm2-bias", str(model.resnets[1].norm2.bias.tolist()),
            "--param-resnets.1-norm2-weight", str(model.resnets[1].norm2.weight.tolist()),
            "--param-resnets.1-conv1-bias", str(model.resnets[1].conv1.bias.tolist()),
            "--param-resnets.1-conv1-weight", str(model.resnets[1].conv1.weight.tolist()),
            "--param-resnets.1-norm1-bias", str(model.resnets[1].norm1.bias.tolist()),
            "--param-resnets.1-norm1-weight", str(model.resnets[1].norm1.weight.tolist()),
            "--param-attentions.0-to_out.0-bias", str(model.attentions[0].to_out[0].bias.tolist()),
            "--param-attentions.0-to_out.0-weight", str(model.attentions[0].to_out[0].weight.tolist()),
            "--param-attentions.0-to_v-weight", str(model.attentions[0].to_v.weight.tolist()),
            "--param-attentions.0-to_v-bias", str(model.attentions[0].to_v.bias.tolist()),
            "--param-attentions.0-to_k-weight", str(model.attentions[0].to_k.weight.tolist()),
            "--param-attentions.0-to_k-bias", str(model.attentions[0].to_k.bias.tolist()),
            "--param-attentions.0-to_q-weight", str(model.attentions[0].to_q.weight.tolist()),
            "--param-attentions.0-to_q-bias", str(model.attentions[0].to_q.bias.tolist()),
            "--param-attentions.0-group_norm-weight", str(model.attentions[0].group_norm.weight.tolist()),
            "--param-attentions.0-group_norm-bias", str(model.attentions[0].group_norm.bias.tolist()),
            "--param-resnets.0-conv2-bias", str(model.resnets[0].conv2.bias.tolist()),
            "--param-resnets.0-conv2-weight", str(model.resnets[0].conv2.weight.tolist()),
            "--param-resnets.0-norm2-bias", str(model.resnets[0].norm2.bias.tolist()),
            "--param-resnets.0-norm2-weight", str(model.resnets[0].norm2.weight.tolist()),
            "--param-resnets.0-conv1-bias", str(model.resnets[0].conv1.bias.tolist()),
            "--param-resnets.0-conv1-weight", str(model.resnets[0].conv1.weight.tolist()),
            "--param-resnets.0-norm1-bias", str(model.resnets[0].norm1.bias.tolist()),
            "--param-resnets.0-norm1-weight", str(model.resnets[0].norm1.weight.tolist()),
        )

        self.assertTensors(actual, [expected])