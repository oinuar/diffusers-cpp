from utils import TestCase
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

class TestTransformersQwen3RotaryEmbedding(TestCase):
    def test_default(self):
        config = Qwen3Config(
            hidden_size=8,
            head_dim=4,
            num_attention_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
        )

        pos_embed = Qwen3RotaryEmbedding(config)

        position_ids = torch.tensor([
            [0, 1, 2, 3],
        ])

        q = torch.randn(
            1,  # batch
            2,  # heads
            position_ids.shape[1],  # sequence
            config.hidden_size // config.num_attention_heads,  # head_dim
        )

        k = torch.randn_like(q)

        cos, sin = pos_embed(q, position_ids)

        expected_q, expected_k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            unsqueeze_dim=1,
        )

        actual_q = self.cli(
            "Qwen3RotaryEmbedding",
            "--head_dim", "4",
            "--rope_theta", "10000",
            "--x", str(q.tolist()),
            "--position_ids", str(position_ids.tolist()),
        )

        actual_k = self.cli(
            "Qwen3RotaryEmbedding",
            "--head_dim", "4",
            "--rope_theta", "10000",
            "--x", str(k.tolist()),
            "--position_ids", str(position_ids.tolist()),
        )

        self.assertTensors(actual_q, [expected_q])
        self.assertTensors(actual_k, [expected_k])
