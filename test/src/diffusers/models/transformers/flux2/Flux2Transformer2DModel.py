from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

class TestNNFlux2Transformer2DModel(TestCase):
    def test_minimal(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,          # inner_dim = 8
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(2, 2),    # sums to head_dim (=4)
            guidance_embeds=False,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])

        img_ids = torch.tensor([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ])

        txt_ids = torch.tensor([
            [
                [0, 0, 0, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
            ]
        ])

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        actual = self.cli(
            "Flux2Transformer2DModel",
            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "1",
            "--num_single_layers", "1",
            "--attention_head_dim", "4",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "2",
            "--axes_dims_rope", "2",
            "--guidance_embeds", "false",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_guidance(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(2, 2),
            guidance_embeds=True,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])
        guidance = torch.tensor([1.0])

        img_ids = torch.tensor([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ])

        txt_ids = torch.tensor([
            [
                [0, 0, 0, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
            ]
        ])

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        actual = self.cli(
            "Flux2Transformer2DModel",

            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "1",
            "--num_single_layers", "1",
            "--attention_head_dim", "4",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "2",
            "--axes_dims_rope", "2",
            "--guidance_embeds", "true",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--guidance", str(guidance.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_multiple_blocks(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=4,
            num_attention_heads=2,          # inner_dim = 8
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(2, 2),    # sums to head_dim (=4)
            guidance_embeds=False,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])

        img_ids = torch.tensor([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ])

        txt_ids = torch.tensor([
            [
                [0, 0, 0, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
            ]
        ])

        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        actual = self.cli(
            "Flux2Transformer2DModel",
            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "2",
            "--num_single_layers", "2",
            "--attention_head_dim", "4",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "2",
            "--axes_dims_rope", "2",
            "--guidance_embeds", "false",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])