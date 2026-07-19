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

            "--param-norm_out-linear-weight", str(model.norm_out.linear.weight.tolist()),
            "--param-proj_out-weight", str(model.proj_out.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_k-weight", str(model.single_transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_q-weight", str(model.single_transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_qkv_mlp_proj-weight", str(model.single_transformer_blocks[0].attn.to_qkv_mlp_proj.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_out-weight", str(model.single_transformer_blocks[0].attn.to_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_out-weight", str(model.transformer_blocks[0].ff_context.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_in-weight", str(model.transformer_blocks[0].ff_context.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_out-weight", str(model.transformer_blocks[0].ff.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_in-weight", str(model.transformer_blocks[0].ff.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_add_out-weight", str(model.transformer_blocks[0].attn.to_add_out.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_v_proj-weight", str(model.transformer_blocks[0].attn.add_v_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_k_proj-weight", str(model.transformer_blocks[0].attn.add_k_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_k-weight", str(model.transformer_blocks[0].attn.norm_added_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_q-weight", str(model.transformer_blocks[0].attn.norm_added_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_out-0-weight", str(model.transformer_blocks[0].attn.to_out[0].weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_q-weight", str(model.transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_q_proj-weight", str(model.transformer_blocks[0].attn.add_q_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_v-weight", str(model.transformer_blocks[0].attn.to_v.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_k-weight", str(model.transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_k-weight", str(model.transformer_blocks[0].attn.to_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_q-weight", str(model.transformer_blocks[0].attn.to_q.weight.tolist()),
            "--param-context_embedder-weight", str(model.context_embedder.weight.tolist()),
            "--param-single_stream_modulation-linear-weight", str(model.single_stream_modulation.linear.weight.tolist()),
            "--param-double_stream_modulation_txt-linear-weight", str(model.double_stream_modulation_txt.linear.weight.tolist()),
            "--param-double_stream_modulation_img-linear-weight", str(model.double_stream_modulation_img.linear.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_2-weight", str(model.time_guidance_embed.timestep_embedder.linear_2.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_1-weight", str(model.time_guidance_embed.timestep_embedder.linear_1.weight.tolist()),
            "--param-x_embedder-weight", str(model.x_embedder.weight.tolist()),
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

            "--param-norm_out-linear-weight", str(model.norm_out.linear.weight.tolist()),
            "--param-proj_out-weight", str(model.proj_out.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_k-weight", str(model.single_transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_q-weight", str(model.single_transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_qkv_mlp_proj-weight", str(model.single_transformer_blocks[0].attn.to_qkv_mlp_proj.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_out-weight", str(model.single_transformer_blocks[0].attn.to_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_out-weight", str(model.transformer_blocks[0].ff_context.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_in-weight", str(model.transformer_blocks[0].ff_context.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_out-weight", str(model.transformer_blocks[0].ff.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_in-weight", str(model.transformer_blocks[0].ff.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_add_out-weight", str(model.transformer_blocks[0].attn.to_add_out.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_v_proj-weight", str(model.transformer_blocks[0].attn.add_v_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_k_proj-weight", str(model.transformer_blocks[0].attn.add_k_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_k-weight", str(model.transformer_blocks[0].attn.norm_added_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_q-weight", str(model.transformer_blocks[0].attn.norm_added_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_out-0-weight", str(model.transformer_blocks[0].attn.to_out[0].weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_q-weight", str(model.transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_q_proj-weight", str(model.transformer_blocks[0].attn.add_q_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_v-weight", str(model.transformer_blocks[0].attn.to_v.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_k-weight", str(model.transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_k-weight", str(model.transformer_blocks[0].attn.to_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_q-weight", str(model.transformer_blocks[0].attn.to_q.weight.tolist()),
            "--param-context_embedder-weight", str(model.context_embedder.weight.tolist()),
            "--param-single_stream_modulation-linear-weight", str(model.single_stream_modulation.linear.weight.tolist()),
            "--param-double_stream_modulation_txt-linear-weight", str(model.double_stream_modulation_txt.linear.weight.tolist()),
            "--param-double_stream_modulation_img-linear-weight", str(model.double_stream_modulation_img.linear.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_1-weight", str(model.time_guidance_embed.timestep_embedder.linear_1.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_2-weight", str(model.time_guidance_embed.timestep_embedder.linear_2.weight.tolist()),
            "--param-time_guidance_embed-guidance_embedder-linear_1-weight", str(model.time_guidance_embed.guidance_embedder.linear_1.weight.tolist()),
            "--param-time_guidance_embed-guidance_embedder-linear_2-weight", str(model.time_guidance_embed.guidance_embedder.linear_2.weight.tolist()),
            "--param-x_embedder-weight", str(model.x_embedder.weight.tolist()),
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

            "--param-norm_out-linear-weight", str(model.norm_out.linear.weight.tolist()),
            "--param-proj_out-weight", str(model.proj_out.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_k-weight", str(model.single_transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-norm_q-weight", str(model.single_transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_qkv_mlp_proj-weight", str(model.single_transformer_blocks[0].attn.to_qkv_mlp_proj.weight.tolist()),
            "--param-single_transformer_blocks-0-attn-to_out-weight", str(model.single_transformer_blocks[0].attn.to_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_out-weight", str(model.transformer_blocks[0].ff_context.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff_context-linear_in-weight", str(model.transformer_blocks[0].ff_context.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_out-weight", str(model.transformer_blocks[0].ff.linear_out.weight.tolist()),
            "--param-transformer_blocks-0-ff-linear_in-weight", str(model.transformer_blocks[0].ff.linear_in.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_add_out-weight", str(model.transformer_blocks[0].attn.to_add_out.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_v_proj-weight", str(model.transformer_blocks[0].attn.add_v_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_k_proj-weight", str(model.transformer_blocks[0].attn.add_k_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_k-weight", str(model.transformer_blocks[0].attn.norm_added_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_added_q-weight", str(model.transformer_blocks[0].attn.norm_added_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_out-0-weight", str(model.transformer_blocks[0].attn.to_out[0].weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_q-weight", str(model.transformer_blocks[0].attn.norm_q.weight.tolist()),
            "--param-transformer_blocks-0-attn-add_q_proj-weight", str(model.transformer_blocks[0].attn.add_q_proj.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_v-weight", str(model.transformer_blocks[0].attn.to_v.weight.tolist()),
            "--param-transformer_blocks-0-attn-norm_k-weight", str(model.transformer_blocks[0].attn.norm_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_k-weight", str(model.transformer_blocks[0].attn.to_k.weight.tolist()),
            "--param-transformer_blocks-0-attn-to_q-weight", str(model.transformer_blocks[0].attn.to_q.weight.tolist()),
            "--param-single_transformer_blocks-1-attn-to_out-weight", str(model.single_transformer_blocks[1].attn.to_out.weight.tolist()),
            "--param-single_transformer_blocks-1-attn-norm_k-weight", str(model.single_transformer_blocks[1].attn.norm_k.weight.tolist()),
            "--param-single_transformer_blocks-1-attn-norm_q-weight", str(model.single_transformer_blocks[1].attn.norm_q.weight.tolist()),
            "--param-single_transformer_blocks-1-attn-to_qkv_mlp_proj-weight", str(model.single_transformer_blocks[1].attn.to_qkv_mlp_proj.weight.tolist()),
            "--param-transformer_blocks-1-ff_context-linear_out-weight", str(model.transformer_blocks[1].ff_context.linear_out.weight.tolist()),
            "--param-transformer_blocks-1-ff_context-linear_in-weight", str(model.transformer_blocks[1].ff_context.linear_in.weight.tolist()),
            "--param-transformer_blocks-1-ff-linear_out-weight", str(model.transformer_blocks[1].ff.linear_out.weight.tolist()),
            "--param-transformer_blocks-1-ff-linear_in-weight", str(model.transformer_blocks[1].ff.linear_in.weight.tolist()),
            "--param-transformer_blocks-1-attn-to_add_out-weight", str(model.transformer_blocks[1].attn.to_add_out.weight.tolist()),
            "--param-transformer_blocks-1-attn-add_v_proj-weight", str(model.transformer_blocks[1].attn.add_v_proj.weight.tolist()),
            "--param-transformer_blocks-1-attn-add_k_proj-weight", str(model.transformer_blocks[1].attn.add_k_proj.weight.tolist()),
            "--param-transformer_blocks-1-attn-norm_added_k-weight", str(model.transformer_blocks[1].attn.norm_added_k.weight.tolist()),
            "--param-transformer_blocks-1-attn-norm_added_q-weight", str(model.transformer_blocks[1].attn.norm_added_q.weight.tolist()),
            "--param-transformer_blocks-1-attn-to_out-0-weight", str(model.transformer_blocks[1].attn.to_out[0].weight.tolist()),
            "--param-transformer_blocks-1-attn-norm_q-weight", str(model.transformer_blocks[1].attn.norm_q.weight.tolist()),
            "--param-transformer_blocks-1-attn-add_q_proj-weight", str(model.transformer_blocks[1].attn.add_q_proj.weight.tolist()),
            "--param-transformer_blocks-1-attn-to_v-weight", str(model.transformer_blocks[1].attn.to_v.weight.tolist()),
            "--param-transformer_blocks-1-attn-norm_k-weight", str(model.transformer_blocks[1].attn.norm_k.weight.tolist()),
            "--param-transformer_blocks-1-attn-to_k-weight", str(model.transformer_blocks[1].attn.to_k.weight.tolist()),
            "--param-transformer_blocks-1-attn-to_q-weight", str(model.transformer_blocks[1].attn.to_q.weight.tolist()),
            "--param-context_embedder-weight", str(model.context_embedder.weight.tolist()),
            "--param-single_stream_modulation-linear-weight", str(model.single_stream_modulation.linear.weight.tolist()),
            "--param-double_stream_modulation_txt-linear-weight", str(model.double_stream_modulation_txt.linear.weight.tolist()),
            "--param-double_stream_modulation_img-linear-weight", str(model.double_stream_modulation_img.linear.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_2-weight", str(model.time_guidance_embed.timestep_embedder.linear_2.weight.tolist()),
            "--param-time_guidance_embed-timestep_embedder-linear_1-weight", str(model.time_guidance_embed.timestep_embedder.linear_1.weight.tolist()),
            "--param-x_embedder-weight", str(model.x_embedder.weight.tolist()),
        )

        self.assertTensors(actual, [expected])