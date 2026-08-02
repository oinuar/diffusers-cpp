from utils import TestCase
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3RotaryEmbedding

class TestTransformersQwen3DecoderLayer(TestCase):
    def test(self):
        config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            _attn_implementation="sdpa"
        )

        model = Qwen3DecoderLayer(config, layer_idx=0)
        rotary_emb = Qwen3RotaryEmbedding(config)

        model.self_attn.is_causal = False

        hidden_states = torch.randn(1, 4, 8)
        position_ids = torch.arange(4).unsqueeze(0)

        position_embeddings = rotary_emb(
            hidden_states,
            position_ids,
        )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = model(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )

        actual = self.cli(
            "Qwen3DecoderLayer",
            "--hidden_size", "8",
            "--intermediate_size", "16",
            "--num_attention_heads", "2",
            "--num_key_value_heads", "2",
            "--max_position_embeddings", "32",
            "--layer_idx", "0",
            "--hidden_states", str(hidden_states.tolist()),
            "--position_ids", str(position_ids.tolist()),
            *self.params(model)
        )

        self.assertTensors(actual, [expected])
