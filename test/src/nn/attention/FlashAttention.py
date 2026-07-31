from utils import TestCase
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlashAttention(TestCase):
    def test_one(self):
        q = torch.tensor([[[[1., 0.]]]])   # (1,1,1,2)
        k = torch.tensor([[[[1., 0.]]]])
        v = torch.tensor([[[[3., 5.]]]])

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_two(self):
        q = torch.tensor([[[[1.,0.],
                            [1.,0.]]]])

        k = torch.tensor([[[[1.,0.],
                            [0.,1.]]]])

        v = torch.tensor([[[[1.,0.],
                            [0.,1.]]]])

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_multi_head(self):
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

    def test_multi_head_long_sequence(self):
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])


    def test_batch_multi_head(self):
        q = torch.randn(3, 2, 5, 8)
        k = torch.randn(3, 2, 5, 8)
        v = torch.randn(3, 2, 5, 8)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])


    def test_single_head_long_sequence(self):
        # Important: catches accidental transpose when H=1
        q = torch.randn(1, 1, 8, 16)
        k = torch.randn(1, 1, 8, 16)
        v = torch.randn(1, 1, 8, 16)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])


    def test_many_heads(self):
        q = torch.randn(1, 8, 3, 32)
        k = torch.randn(1, 8, 3, 32)
        v = torch.randn(1, 8, 3, 32)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])

        def test_different_qkv_values(self):
            # Makes sure we are not accidentally returning V or mixing heads
            q = torch.tensor(
                [[[
                    [10., 0.],
                    [0., 10.]
                ]]]
            )

            k = torch.tensor(
                [[[
                    [10., 0.],
                    [0., 10.]
                ]]]
            )

            v = torch.tensor(
                [[[
                    [1., 2.],
                    [3., 4.]
                ]]]
            )

            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                expected = F.scaled_dot_product_attention(q, k, v)

            actual = self.cli(
                "FlashAttention",
                "--q", str(q.tolist()),
                "--k", str(k.tolist()),
                "--v", str(v.tolist()),
            )

            self.assertTensors(actual, [expected])

    def test_attention_mask(self):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)

        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :, 2:] = -1e4

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask
            )

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
            "--mask", str(mask.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_attention_mask_multi_batch_head(self):
        q = torch.randn(3, 4, 5, 8)
        k = torch.randn(3, 4, 5, 8)
        v = torch.randn(3, 4, 5, 8)

        mask = torch.zeros(3, 1, 5, 5)
        mask[:, :, :, -1] = -1e4

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask
            )

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
            "--mask", str(mask.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_gqa(self):
        q = torch.randn(1, 4, 5, 8)
        k = torch.randn(1, 2, 5, 8)
        v = torch.randn(1, 2, 5, 8)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        actual = self.cli(
            "FlashAttention",
            "--q", str(q.tolist()),
            "--k", str(k.tolist()),
            "--v", str(v.tolist()),
        )

        self.assertTensors(actual, [expected])
