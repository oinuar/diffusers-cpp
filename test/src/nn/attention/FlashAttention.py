from utils import TestCase
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlashAttention(TestCase):
    def test_multi_head(self):
        # This standalone test cannot be ran on more than one device 
        # beacuse GGML hard-asserts that flash attention inputs must
        # be S(2). Here inputs are R.
        if self.n_devices() > 1:
            print("Skipped because N_DEVICES > 1")
            return

        q = torch.randn(1, 2, 2, 4)
        k = torch.randn(1, 2, 2, 4)
        v = torch.randn(1, 2, 2, 4)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])
