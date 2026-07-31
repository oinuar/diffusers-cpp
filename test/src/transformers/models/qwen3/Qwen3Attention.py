from utils import TestCase
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding

class TestTransformersQwen3Attention(TestCase):
    def test(self):
        config = Qwen3Config(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=128,
            attention_dropout=0.0,
            _attn_implementation="sdpa"
        )

        model = Qwen3Attention(config, layer_idx=0)
        rotary_emb = Qwen3RotaryEmbedding(config)

        model.is_causal = False

        hidden_states = torch.randn(1, 4, 16)
        position_ids = torch.arange(4).unsqueeze(0)

        position_embeddings = rotary_emb(
            hidden_states,
            position_ids,
        )

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = model(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]

        actual = self.cli(
            "Qwen3Attention",
            "--hidden_size", "16",
            "--num_attention_heads", "4",
            "--num_key_value_heads", "2",
            "--head_dim", "4",
            "--hidden_states", str(hidden_states.tolist()),
            "--position_ids", str(position_ids.tolist()),
            "--layer_idx", "0",
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
