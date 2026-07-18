from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2TransformerBlock(TestCase):
    def test_default(self):
        model = Flux2TransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
        )

        hidden_states = torch.randn(1, 1, 8)
        encoder_hidden_states = torch.randn(1, 1, 8)

        temb_mod_img = torch.randn(1, 1, 48)
        temb_mod_txt = torch.randn(1, 1, 48)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",

            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),

            "--param-attn-to_q-weight", str(model.attn.to_q.weight.tolist()),
            "--param-attn-to_k-weight", str(model.attn.to_k.weight.tolist()),
            "--param-attn-to_v-weight", str(model.attn.to_v.weight.tolist()),

            "--param-attn-norm_q-weight", str(model.attn.norm_q.weight.tolist()),
            "--param-attn-norm_k-weight", str(model.attn.norm_k.weight.tolist()),

            "--param-attn-to_out-0-weight", str(model.attn.to_out[0].weight.tolist()),

            "--param-attn-norm_added_q-weight", str(model.attn.norm_added_q.weight.tolist()),
            "--param-attn-norm_added_k-weight", str(model.attn.norm_added_k.weight.tolist()),
            "--param-attn-add_q_proj-weight", str(model.attn.add_q_proj.weight.tolist()),
            "--param-attn-add_k_proj-weight", str(model.attn.add_k_proj.weight.tolist()),
            "--param-attn-add_v_proj-weight", str(model.attn.add_v_proj.weight.tolist()),
            "--param-attn-to_add_out-weight", str(model.attn.to_add_out.weight.tolist()),

            "--param-ff-linear_in-weight", str(model.ff.linear_in.weight.tolist()),
            "--param-ff-linear_out-weight", str(model.ff.linear_out.weight.tolist()),

            "--param-ff_context-linear_in-weight", str(model.ff_context.linear_in.weight.tolist()),
            "--param-ff_context-linear_out-weight", str(model.ff_context.linear_out.weight.tolist()),
        )

        self.assertTensors(actual, list(expected))

    def test_bias(self):
        model = Flux2TransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
            bias=True,
        )

        hidden_states = torch.randn(1, 1, 8)
        encoder_hidden_states = torch.randn(1, 1, 8)

        temb_mod_img = torch.randn(1, 1, 48)
        temb_mod_txt = torch.randn(1, 1, 48)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",
            "--bias", "true",

            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),

            "--param-attn-to_q-weight", str(model.attn.to_q.weight.tolist()),
            "--param-attn-to_q-bias", str(model.attn.to_q.bias.tolist()),
            "--param-attn-to_k-weight", str(model.attn.to_k.weight.tolist()),
            "--param-attn-to_k-bias", str(model.attn.to_k.bias.tolist()),
            "--param-attn-to_v-weight", str(model.attn.to_v.weight.tolist()),
            "--param-attn-to_v-bias", str(model.attn.to_v.bias.tolist()),

            "--param-attn-norm_q-weight", str(model.attn.norm_q.weight.tolist()),
            "--param-attn-norm_k-weight", str(model.attn.norm_k.weight.tolist()),

            "--param-attn-to_out-0-weight", str(model.attn.to_out[0].weight.tolist()),
            "--param-attn-to_out-0-bias", str(model.attn.to_out[0].bias.tolist()),

            "--param-attn-norm_added_q-weight", str(model.attn.norm_added_q.weight.tolist()),
            "--param-attn-norm_added_k-weight", str(model.attn.norm_added_k.weight.tolist()),
            "--param-attn-add_q_proj-weight", str(model.attn.add_q_proj.weight.tolist()),
            "--param-attn-add_q_proj-bias", str(model.attn.add_q_proj.bias.tolist()),
            "--param-attn-add_k_proj-weight", str(model.attn.add_k_proj.weight.tolist()),
            "--param-attn-add_k_proj-bias", str(model.attn.add_k_proj.bias.tolist()),
            "--param-attn-add_v_proj-weight", str(model.attn.add_v_proj.weight.tolist()),
            "--param-attn-add_v_proj-bias", str(model.attn.add_v_proj.bias.tolist()),
            "--param-attn-to_add_out-weight", str(model.attn.to_add_out.weight.tolist()),
            "--param-attn-to_add_out-bias", str(model.attn.to_add_out.bias.tolist()),

            "--param-ff-linear_in-weight", str(model.ff.linear_in.weight.tolist()),
            "--param-ff-linear_in-bias", str(model.ff.linear_in.bias.tolist()),
            "--param-ff-linear_out-weight", str(model.ff.linear_out.weight.tolist()),
            "--param-ff-linear_out-bias", str(model.ff.linear_out.bias.tolist()),

            "--param-ff_context-linear_in-weight", str(model.ff_context.linear_in.weight.tolist()),
            "--param-ff_context-linear_in-bias", str(model.ff_context.linear_in.bias.tolist()),
            "--param-ff_context-linear_out-weight", str(model.ff_context.linear_out.weight.tolist()),
            "--param-ff_context-linear_out-bias", str(model.ff_context.linear_out.bias.tolist()),
        )

        self.assertTensors(actual, list(expected))

    def test_multiple_tokens(self):
        model = Flux2TransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
        )

        hidden_states = torch.randn(1, 4, 8)
        encoder_hidden_states = torch.randn(1, 3, 8)

        temb_mod_img = torch.randn(1, 1, 48)
        temb_mod_txt = torch.randn(1, 1, 48)

        expected = model(
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
        )

        actual = self.cli(
            "Flux2TransformerBlock",
            "--dim", "8",
            "--num_attention_heads", "2",
            "--attention_head_dim", "4",

            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--temb_mod_img", str(temb_mod_img.tolist()),
            "--temb_mod_txt", str(temb_mod_txt.tolist()),

            "--param-attn-to_q-weight", str(model.attn.to_q.weight.tolist()),
            "--param-attn-to_k-weight", str(model.attn.to_k.weight.tolist()),
            "--param-attn-to_v-weight", str(model.attn.to_v.weight.tolist()),

            "--param-attn-norm_q-weight", str(model.attn.norm_q.weight.tolist()),
            "--param-attn-norm_k-weight", str(model.attn.norm_k.weight.tolist()),

            "--param-attn-to_out-0-weight", str(model.attn.to_out[0].weight.tolist()),

            "--param-attn-norm_added_q-weight", str(model.attn.norm_added_q.weight.tolist()),
            "--param-attn-norm_added_k-weight", str(model.attn.norm_added_k.weight.tolist()),
            "--param-attn-add_q_proj-weight", str(model.attn.add_q_proj.weight.tolist()),
            "--param-attn-add_k_proj-weight", str(model.attn.add_k_proj.weight.tolist()),
            "--param-attn-add_v_proj-weight", str(model.attn.add_v_proj.weight.tolist()),
            "--param-attn-to_add_out-weight", str(model.attn.to_add_out.weight.tolist()),

            "--param-ff-linear_in-weight", str(model.ff.linear_in.weight.tolist()),
            "--param-ff-linear_out-weight", str(model.ff.linear_out.weight.tolist()),

            "--param-ff_context-linear_in-weight", str(model.ff_context.linear_in.weight.tolist()),
            "--param-ff_context-linear_out-weight", str(model.ff_context.linear_out.weight.tolist()),
        )

        self.assertTensors(actual, list(expected))