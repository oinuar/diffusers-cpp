#!/usr/bin/env python3
"""Comprehensive unittest suite for Tensor C++ class via tensor-cli subprocess.

Every test case:
  1. Creates a PyTorch tensor on CPU and runs the same operation.
  2. Calls tensor-cli via subprocess to run the equivalent operation.
  3. Compares the C++ result against PyTorch's reference output element-wise
     with an absolute tolerance (1e-4 for float32).

Each test exercises exactly one branch/operator of the Tensor CLI program,
covering 1D, 2D, 3D and 4D tensor ranks.
"""

import math
import re
import subprocess
import unittest
import torch
import os
import ast


def cli(*args: str) -> torch.Tensor:
    cli = os.environ.get("TENSOR_CLI", "tensor-cli")

    result = subprocess.run([cli, *args], capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        raise RuntimeError(
            f"tensor-cli failed (rc={result.returncode}):\n{result.stderr}"
        )

    return torch.tensor(ast.literal_eval(result.stdout), dtype=torch.float32)

def verify(
    self,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol))


# ---------------------------------------------------------------------------
# Factories  (empty, scalar, zeros, ones, arange)
# ---------------------------------------------------------------------------

class TestFactoryMethods(unittest.TestCase):
    def test_scalar(self):
        pt = torch.tensor([42.0])  # scalar tensor (0-dim)
        expected = pt

        result = cli("scalar", "--value", "42.0")
        verify(self, result, expected)

    def test_scalar_negative(self):
        pt = torch.tensor([-3.14])
        expected = pt

        result = cli("scalar", "--value", "-3.14")
        verify(self, result, expected)

    def test_zeros_1d(self):
        pt = torch.zeros(5)
        expected = pt

        result = cli("zeros", "--shape", "(5)")
        verify(self, result, expected)

    def test_zeros_2d(self):
        pt = torch.zeros(2, 3)
        expected = pt

        result = cli("zeros", "--shape", "(2, 3)")
        verify(self, result, expected)

    def test_zeros_3d(self):
        pt = torch.zeros(2, 3, 4)
        expected = pt

        result = cli("zeros", "--shape", "(2, 3, 4)")
        verify(self, result, expected)

    def test_zeros_4d(self):
        pt = torch.zeros(1, 2, 3, 4)
        expected = pt

        result = cli("zeros", "--shape", "(1, 2, 3, 4)")
        verify(self, result, expected)

    def test_ones_1d(self):
        pt = torch.ones(4)
        expected = pt

        result = cli("ones", "--shape", "(4)")
        verify(self, result, expected)

    def test_ones_2d(self):
        pt = torch.ones(3, 4)
        expected = pt

        result = cli("ones", "--shape", "(3, 4)")
        verify(self, result, expected)

    def test_ones_3d(self):
        pt = torch.ones(2, 2, 3)
        expected = pt

        result = cli("ones", "--shape", "(2, 2, 3)")
        verify(self, result, expected)

    def test_ones_4d(self):
        # Actual rank-4: (1, 2, 3, 4) — all four dims specified explicitly.
        pt = torch.ones(1, 2, 3, 4)
        expected = pt

        result = cli("ones", "--shape", "(1, 2, 3, 4)")
        verify(self, result, expected)

    def test_arange_default(self):
        pt = torch.arange(0.0, 5.0)
        expected = pt

        result = cli("arange", "--start", "0.0", "--stop", "5.0", "--step", "1.0")
        verify(self, result, expected)

    def test_arange_step(self):
        pt = torch.arange(0.0, 3.0, 0.5)
        expected = pt

        result = cli("arange", "--start", "0.0", "--stop", "3.0", "--step", "0.5")
        verify(self, result, expected)

    def test_arange_negative_start(self):
        pt = torch.arange(-2.0, 2.0)
        expected = pt

        result = cli("arange", "--start", "-2.0", "--stop", "2.0", "--step", "1.0")
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Shape operations  (reshape, permute, squeeze, unsqueeze)
# ---------------------------------------------------------------------------

class TestShapeOperations(unittest.TestCase):
    def test_reshape_2d_to_1d(self):
        """reshape (2,3) -> (6)"""
        data = list(range(1, 7))  # [1..6], 6 elements
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.reshape(6)

        result = cli("reshape", "--this", str(pt.tolist()), "--shape", "(6)")
        verify(self, result, expected)

    def test_reshape_1d_to_2d(self):
        """reshape (6) -> (2,3)"""
        data = list(range(1, 7))  # [1..6], 6 elements
        pt = torch.tensor(data).float().reshape(6)
        expected = pt.reshape(2, 3)

        result = cli("reshape", "--this", str(pt.tolist()), "--shape", "(2, 3)")
        verify(self, result, expected)

    def test_reshape_with_infer(self):
        """reshape (4,) -> (2, 2)"""
        data = list(range(1, 5))  # [1..4], 4 elements
        pt = torch.tensor(data).float().reshape(4)
        expected = pt.reshape(2, 2)

        result = cli("reshape", "--this", str(pt.tolist()), "--shape", "(2, 2)")
        verify(self, result, expected)

    def test_reshape_3d_to_1d(self):
        """reshape (2,3,4) -> (24) — rank-3 tensor flattened to rank-1"""
        data = list(range(1, 25))  # [1..24]
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.reshape(24)

        result = cli("reshape", "--this", str(pt.tolist()), "--shape", "(24)")
        verify(self, result, expected)

    def test_reshape_4d_to_2d(self):
        """reshape (1,2,3,4) -> (6,4) — rank-4 tensor reshaped to rank-2"""
        data = list(range(1, 25))  # [1..24]
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.reshape(6, 4)

        result = cli("reshape", "--this", str(pt.tolist()), "--shape", "(6, 4)")
        verify(self, result, expected)

    def test_permute_2d(self):
        """permute(1,0) on (3,4) — swap dimensions"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt.permute(1, 0)

        result = cli("permute", "--this", str(pt.tolist()), "--order", "(1, 0)")
        verify(self, result, expected)

    def test_permute_3d(self):
        """permute(2,1,0) on rank-3 tensor (2,3,4) — reverse dimensions"""
        data = list(range(24))  # [0..23]
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.permute(2, 1, 0)

        result = cli("permute", "--this", str(pt.tolist()), "--order", "(2, 1, 0)")
        verify(self, result, expected)

    def test_squeeze(self):
        """squeeze dim=0 on (1,N) -> (N)"""
        data = list(range(5))  # [0..4], 5 elements
        pt = torch.tensor(data).float().reshape(1, 5)
        expected = pt.squeeze(0)

        result = cli("squeeze", "--this", str(pt.tolist()), "--dim", "0")
        verify(self, result, expected)

    def test_squeeze_dim1(self):
        """squeeze dim=1 on rank-3 tensor (N,1,P) -> (N,P)"""
        data = list(range(8))  # 2*1*4 = 8 elements
        pt = torch.tensor(data).float().reshape(2, 1, 4)
        expected = pt.squeeze(1)

        result = cli("squeeze", "--this", str(pt.tolist()), "--dim", "1")
        verify(self, result, expected)

    def test_squeeze_3d(self):
        """squeeze dim=0 on rank-3 tensor (1,M,P,Q)"""
        data = list(range(48))  # [0..47], shape (1,6,8)=48
        pt = torch.tensor(data).float().reshape(1, 6, 8)
        expected = pt.squeeze(0)

        result = cli("squeeze", "--this", str(pt.tolist()), "--dim", "0")
        verify(self, result, expected)

    def test_unsqueeze(self):
        """unsqueeze dim=0 on (N,) -> (1,N)"""
        data = list(range(3))  # [0, 1, 2], 3 elements
        pt = torch.tensor(data).float().reshape(3)
        expected = pt.unsqueeze(0)

        result = cli("unsqueeze", "--this", str(pt.tolist()), "--dim", "0")
        verify(self, result, expected)

    def test_unsqueeze_2d(self):
        """unsqueeze dim=1 on (N,M) -> (N,1,M)"""
        data = list(range(6))
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.unsqueeze(1)

        result = cli("unsqueeze", "--this", str(pt.tolist()), "--dim", "1")
        verify(self, result, expected)

    def test_flatten(self):
        """flatten(start_dim=1, end_dim=2) on (2,3) -> (2,3)"""
        data = list(range(6))  # 6 elements
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = pt.flatten(start_dim=1, end_dim=2)

        result = cli("flatten", "--this", str(pt.tolist()), "--start_dim", "1", "--end_dim", "2")
        verify(self, result, expected)

    def test_flatten_4d(self):
        """flatten(start_dim=1, end_dim=2) on rank-4 tensor (1,2,3,4)"""
        data = list(range(24))  # [0..23], shape (1,2,3,4)=24
        pt = torch.tensor(data).float().reshape(1, 2, 3, 4)
        expected = pt.flatten(start_dim=1, end_dim=2)

        result = cli("flatten", "--this", str(pt.tolist()), "--start_dim", "1", "--end_dim", "2")
        verify(self, result, expected)

    def test_unflatten(self):
        """unflatten dim=0 shape=(2,3) on (6,)"""
        data = [float(i) for i in range(6)]
        pt = torch.tensor(data).reshape(6)
        expected = pt.unflatten(0, (2, 3))

        result = cli("unflatten", "--this", str(pt.tolist()), "--dim", "0", "--shape", "(2, 3)")
        verify(self, result, expected)

    def test_unflatten_3d(self):
        """unflatten dim=0 shape=(3,4) on (12,)"""
        data = [float(i) for i in range(12)]
        pt = torch.tensor(data).reshape(12)
        expected = pt.unflatten(0, (3, 4))

        result = cli("unflatten", "--this", str(pt.tolist()), "--dim", "0", "--shape", "(3, 4)")
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Slicing operations  (narrow/contiguous via [], expand, index [], slice)
# ---------------------------------------------------------------------------

class TestSlicingOperations(unittest.TestCase):
    def test_contiguous_1d(self):
        """contiguous() on already-contiguous tensor"""
        data = list(range(4))  # [0..3], 4 elements
        pt = torch.tensor(data).float().reshape(2, 2)
        expected = pt.contiguous()

        result = cli("contiguous", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_contiguous_2d(self):
        """contiguous() on a permuted (non-contiguous) tensor"""
        data = list(range(12))
        pt = torch.tensor(data).float().reshape(3, 4).transpose(0, 1)
        expected = pt.contiguous()

        result = cli("contiguous", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_contiguous_3d(self):
        """contiguous() on a permuted rank-3 tensor"""
        data = list(range(24))  # [0..23]
        pt = torch.tensor(data).float().reshape(2, 3, 4).transpose(0, 1)
        expected = pt.contiguous()

        result = cli("contiguous", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_narrow(self):
        """narrow(dim=1, start=1, length=2) on (2,4)"""
        data = list(range(8))  # [[0..3],[4..7]]
        pt = torch.tensor(data).float().reshape(2, 4)
        expected = pt.narrow(1, 1, 2)

        result = cli("narrow", "--this", str(pt.tolist()), "--dim", "1", "--start", "1", "--length", "2")
        verify(self, result, expected)

    def test_narrow_3d(self):
        """narrow(dim=0, start=1, length=1) on rank-3 tensor (2,3,4)"""
        data = list(range(24))  # [0..23], shape (2,3,4)=24
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt.narrow(0, 1, 1)

        result = cli("narrow", "--this", str(pt.tolist()), "--dim", "0", "--start", "1", "--length", "1")
        verify(self, result, expected)

    def test_expand(self):
        """expand (1,) -> (2,) — broadcasting"""
        data = [1.0]
        pt = torch.tensor(data)  # shape (1,)
        expected = pt.expand(2)  # shape (2,), broadcasts single element

        result = cli("expand", "--this", str(pt.tolist()), "--new-shape", "(2)")
        verify(self, result, expected)

    def test_expand_2d(self):
        """expand (1,2) -> (3,2) — broadcasting on rank-2 tensor"""
        data = [1.0, 2.0]
        pt = torch.tensor(data).reshape(1, 2)
        expected = pt.expand(3, 2)

        result = cli("expand", "--this", str(pt.tolist()), "--new-shape", "(3, 2)")
        verify(self, result, expected)

    def test_index_single(self):
        """index [1] on (4,)"""
        data = list(range(4))  # [0, 1, 2, 3]
        pt = torch.tensor(data).float()
        expected = pt[1]

        result = cli("index", "--this", str(pt.tolist()), "--index", "1")
        verify(self, result, expected)

    def test_index_first(self):
        """index [0] on (3,)"""
        data = list(range(3))  # [0, 1, 2]
        pt = torch.tensor(data).float()
        expected = pt[0]

        result = cli("index", "--this", str(pt.tolist()), "--index", "0")
        verify(self, result, expected)

    def test_index_2d(self):
        """index [1] on (2,4)"""
        data = list(range(8))  # [[0..3],[4..7]]
        pt = torch.tensor(data).float().reshape(2, 4)
        expected = pt[1]

        result = cli("index", "--this", str(pt.tolist()), "--index", "1")
        verify(self, result, expected)

    def test_slice_all(self):
        """slice [:] — full slice"""
        data = list(range(3))  # [0, 1, 2]
        pt = torch.tensor(data).float()
        expected = pt[:]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[:]")
        verify(self, result, expected)

    def test_slice_range(self):
        """slice [1:3]"""
        data = list(range(5))  # [0, 1, 2, 3, 4]
        pt = torch.tensor(data).float()
        expected = pt[1:3]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[1:3]")
        verify(self, result, expected)

    def test_slice_with_step(self):
        """slice [::2] — every other element"""
        data = list(range(5))  # [0, 1, 2, 3, 4]
        pt = torch.tensor(data).float()
        expected = pt[::2]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[::2]")
        verify(self, result, expected)

    def test_slice_2d(self):
        """slice [:, 1:3] on (3,4)"""
        data = list(range(12))  # [[0..3],[4..7],[8..11]]
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[:, 1:3]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[:, 1:3]")
        verify(self, result, expected)

    def test_slice_2d_step(self):
        """slice [::2,:] on (3,4)"""
        data = list(range(12))  # [[0..3],[4..7],[8..11]]
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[::2, :]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[::2, :]")
        verify(self, result, expected)

    def test_slice_3d(self):
        """slice [:,:,1:] on rank-3 tensor (2,3,4)"""
        data = list(range(24))  # [0..23]
        pt = torch.tensor(data).float().reshape(2, 3, 4)
        expected = pt[:, :, 1:]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[:, :, 1:]")
        verify(self, result, expected)

    def test_slice_newaxis(self):
        """slice [:, None] — insert new axis (dim=1)"""
        data = list(range(3))  # [0, 1, 2]
        pt = torch.tensor(data).float()
        expected = pt[:, None]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[:, None]")
        verify(self, result, expected)

    def test_slice_newaxis_2d(self):
        """slice [None,:,None] — insert new axes"""
        data = list(range(12))  # [[0..3],[4..7],[8..11]]
        pt = torch.tensor(data).float().reshape(3, 4)
        expected = pt[None, :, None]

        result = cli("slice", "--this", str(pt.tolist()), "--slice", "[None, :, None]")
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Binary operators  (add/sub/mul/div between two tensors)
# ---------------------------------------------------------------------------

class TestBinaryOperators(unittest.TestCase):
    def test_add_1d(self):
        """[1] + [4]"""
        a_vals = [1.0]
        b_vals = [4.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs + rhs

        result = cli("add", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_add_2d(self):
        """[[1],[3]] + [[5],[7]]"""
        a_vals = [[1.0], [3.0]]
        b_vals = [[5.0], [7.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs + rhs

        result = cli("add", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_add_3d(self):
        """[2] + [2] — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) for v in data]
        b_vals = [float(v) * 0.5 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs + rhs

        result = cli("add", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_sub_1d(self):
        """[10] - [1]"""
        a_vals = [10.0]
        b_vals = [1.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs - rhs

        result = cli("sub", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_sub_2d(self):
        """[[5],[7]] - [[1],[3]]"""
        a_vals = [[5.0], [7.0]]
        b_vals = [[1.0], [3.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs - rhs

        result = cli("sub", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_sub_3d(self):
        """[2] - [2] — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) for v in data]
        b_vals = [float(v) * 0.5 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs - rhs

        result = cli("sub", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_mul_1d(self):
        """[2] * [4]"""
        a_vals = [2.0]
        b_vals = [4.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs

        result = cli("mul", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_mul_2d(self):
        """[[2],[3]] * [[4],[5]]"""
        a_vals = [[2.0], [3.0]]
        b_vals = [[4.0], [5.0]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs * rhs

        result = cli("mul", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_mul_3d(self):
        """[2] * [2] — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) for v in data]
        b_vals = [float(v) + 1.0 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs * rhs

        result = cli("mul", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_div_1d(self):
        """[10.] / [2.]"""
        a_vals = [10.0]
        b_vals = [2.0]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs / rhs

        result = cli("div", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_div_2d(self):
        """[[10],[30]] / [[2.5],[6.5]]"""
        a_vals = [[10.0], [30.0]]
        b_vals = [[2.5], [6.5]]
        lhs = torch.tensor(a_vals)
        rhs = torch.tensor(b_vals)
        expected = lhs / rhs

        result = cli("div", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)

    def test_div_3d(self):
        """[2] / [2] — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) + 1.0 for v in data]
        b_vals = [float(v) + 2.0 for v in data]
        lhs = torch.tensor(a_vals).reshape(2, 3, 4)
        rhs = torch.tensor(b_vals).reshape(2, 3, 4)
        expected = lhs / rhs

        result = cli("div", "--lhs", str(lhs.tolist()), "--rhs", str(rhs.tolist()))
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Scalar binary operators  (tensor op-scalar and scalar op-tensor)
# ---------------------------------------------------------------------------

class TestScalarBinaryOperators(unittest.TestCase):
    def test_add_scalar_1d(self):
        """[1] + 3"""
        a_vals = [1.0]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0

        result = cli("add_scalar", "--lhs", str(pt.tolist()), "--rhs", "3.0")
        verify(self, result, expected)

    def test_add_scalar_2d(self):
        """[[1],[2]] + 3"""
        a_vals = [[1.0], [2.0]]
        pt = torch.tensor(a_vals)
        expected = pt + 3.0

        result = cli("add_scalar", "--lhs", str(pt.tolist()), "--rhs", "3.0")
        verify(self, result, expected)

    def test_add_scalar_3d(self):
        """tensor + 3.0 — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt + 3.0

        result = cli("add_scalar", "--lhs", str(pt.tolist()), "--rhs", "3.0")
        verify(self, result, expected)

    def test_sub_scalar_1d(self):
        """[10] - 5"""
        a_vals = [10.0]
        pt = torch.tensor(a_vals)
        expected = pt - 5.0

        result = cli("sub_scalar", "--lhs", str(pt.tolist()), "--rhs", "5.0")
        verify(self, result, expected)

    def test_sub_scalar_2d(self):
        """[[10],[20]] - 5"""
        a_vals = [[10.0], [20.0]]
        pt = torch.tensor(a_vals) - 5.0
        expected = pt - 5.0

        result = cli("sub_scalar", "--lhs", str(pt.tolist()), "--rhs", "5.0")
        verify(self, result, expected)

    def test_mul_scalar_1d(self):
        """[2] * 4"""
        a_vals = [2.0]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0

        result = cli("mul_scalar", "--lhs", str(pt.tolist()), "--rhs", "4.0")
        verify(self, result, expected)

    def test_mul_scalar_2d(self):
        """[[2],[3]] * 4"""
        a_vals = [[2.0], [3.0]]
        pt = torch.tensor(a_vals)
        expected = pt * 4.0

        result = cli("mul_scalar", "--lhs", str(pt.tolist()), "--rhs", "4.0")
        verify(self, result, expected)

    def test_mul_scalar_3d(self):
        """tensor * 4.0 — rank-3 tensor"""
        data = list(range(1, 25))  # [1..24], shape (2,3,4)=24
        a_vals = [float(v) for v in data]
        pt = torch.tensor(a_vals).reshape(2, 3, 4)
        expected = pt * 4.0

        result = cli("mul_scalar", "--lhs", str(pt.tolist()), "--rhs", "4.0")
        verify(self, result, expected)

    def test_div_scalar_1d(self):
        """[10.] / 5"""
        a_vals = [10.0]
        pt = torch.tensor(a_vals)
        expected = pt / 5.0

        result = cli("div_scalar", "--lhs", str(pt.tolist()), "--rhs", "5.0")
        verify(self, result, expected)

    def test_div_scalar_2d(self):
        """[[10],[20]] / 5"""
        a_vals = [[10.0], [20.0]]
        pt = torch.tensor(a_vals)
        expected = pt / 5.0

        result = cli("div_scalar", "--lhs", str(pt.tolist()), "--rhs", "5.0")
        verify(self, result, expected)

    def test_scalar_add_1d(self):
        """10 + [1]"""
        b_vals = [1.0]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt

        result = cli("scalar_add", "--lhs", "10.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_add_2d(self):
        """10 + [[1],[2]]"""
        b_vals = [[1.0], [2.0]]
        pt = torch.tensor(b_vals)
        expected = 10.0 + pt

        result = cli("scalar_add", "--lhs", "10.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_sub_1d(self):
        """20 - [3]"""
        b_vals = [3.0]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt

        result = cli("scalar_sub", "--lhs", "20.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_sub_2d(self):
        """20 - [[3],[5]]"""
        b_vals = [[3.0], [5.0]]
        pt = torch.tensor(b_vals)
        expected = 20.0 - pt

        result = cli("scalar_sub", "--lhs", "20.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_mul_1d(self):
        """3 * [4]"""
        b_vals = [4.0]
        pt = torch.tensor(b_vals)
        expected = 3.0 * pt

        result = cli("scalar_mul", "--lhs", "3.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_mul_2d(self):
        """3 * [[4],[6]]"""
        b_vals = [[4.0], [6.0]]
        pt = torch.tensor(b_vals)
        expected = 3.0 * pt

        result = cli("scalar_mul", "--lhs", "3.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_div_1d(self):
        """50 / [5]"""
        b_vals = [5.0]
        pt = torch.tensor(b_vals)
        expected = 50.0 / pt

        result = cli("scalar_div", "--lhs", "50.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)

    def test_scalar_div_2d(self):
        """50 / [[5],[10]]"""
        b_vals = [[5.0], [10.0]]
        pt = torch.tensor(b_vals)
        expected = 50.0 / pt

        result = cli("scalar_div", "--lhs", "50.0", "--rhs", str(pt.tolist()))
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Unary operators  (neg)
# ---------------------------------------------------------------------------

class TestUnaryOperators(unittest.TestCase):
    def test_neg_1d(self):
        """-[1,-2]"""
        data = [1.0, -2.0]
        pt = torch.tensor(data)
        expected = -pt

        result = cli("neg", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_neg_2d(self):
        """-[[1,-2],[3,-4]]"""
        data = [[1.0], [-2.0]]
        pt = torch.tensor(data)
        expected = -pt

        result = cli("neg", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_neg_3d(self):
        """neg on rank-3 tensor (N,M,P,Q)"""
        data = [[[-1.0], [2.0]], [[-3.0], [4.0]]]
        pt = torch.tensor(data)
        expected = -pt

        result = cli("neg", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_neg_zeros_1d(self):
        """-[0]"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = -pt

        result = cli("neg", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_neg_zeros_2d(self):
        """-[[0],[0]]"""
        data = [[0.0], [0.0]]
        pt = torch.tensor(data)
        expected = -pt

        result = cli("neg", "--this", str(pt.tolist()))
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Element-wise functions  (abs, sqrt, exp, log, sin, cos, rsqrt)
# ---------------------------------------------------------------------------

class TestElementwiseFunctions(unittest.TestCase):
    def test_abs_1d(self):
        """abs([-3.5,-1.2])"""
        data = [-3.5]
        pt = torch.tensor(data)
        expected = torch.abs(pt)

        result = cli("abs", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_abs_2d(self):
        """abs([[-3.5,-1.2]])"""
        data = [[-3.5], [-1.2]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)

        result = cli("abs", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_abs_3d(self):
        """abs on rank-3 tensor"""
        data = [[[-3.5], [-1.2]], [[2.0], [-0.5]]]
        pt = torch.tensor(data)
        expected = torch.abs(pt)

        result = cli("abs", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_abs_all_positive(self):
        """abs([1]) — no-op"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.abs(pt)

        result = cli("abs", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sqrt_1d(self):
        """sqrt([4])"""
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)

        result = cli("sqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sqrt_2d(self):
        """sqrt([[4],[9]]"""
        data = [[4.0], [9.0]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)

        result = cli("sqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sqrt_3d(self):
        """sqrt on rank-3 tensor"""
        data = [[[4.0], [9.0]], [[16.0], [25.0]]]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)

        result = cli("sqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sqrt_small(self):
        """sqrt([0.25])"""
        data = [0.25]
        pt = torch.tensor(data)
        expected = torch.sqrt(pt)

        result = cli("sqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_exp_1d(self):
        """exp([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.exp(pt)

        result = cli("exp", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_exp_2d(self):
        """exp([[0],[1]]"""
        data = [[0.0], [1.0]]
        pt = torch.tensor(data)
        expected = torch.exp(pt)

        result = cli("exp", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_exp_negative(self):
        """exp([-1])"""
        data = [-1.0]
        pt = torch.tensor(data)
        expected = torch.exp(pt)

        result = cli("exp", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_log_1d(self):
        """log([1])"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)

        result = cli("log", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_log_2d(self):
        """log([[1],[e]]"""
        data = [[1.0], [math.e]]
        pt = torch.tensor(data)
        expected = torch.log(pt)

        result = cli("log", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_log_values(self):
        """log([10])"""
        data = [10.0]
        pt = torch.tensor(data)
        expected = torch.log(pt)

        result = cli("log", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sin_1d(self):
        """sin([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.sin(pt)

        result = cli("sin", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sin_2d(self):
        """sin([[0],[pi/2]]"""
        data = [[0.0], [math.pi / 2]]
        pt = torch.tensor(data)
        expected = torch.sin(pt)

        result = cli("sin", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_cos_1d(self):
        """cos([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)

        result = cli("cos", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_cos_2d(self):
        """cos([[0],[pi]]"""
        data = [[0.0], [math.pi]]
        pt = torch.tensor(data)
        expected = torch.cos(pt)

        result = cli("cos", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_cos_zero(self):
        """cos([0])"""
        data = [0.0]
        pt = torch.tensor(data)
        expected = torch.cos(pt)

        result = cli("cos", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_rsqrt_1d(self):
        """rsqrt([4])"""
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)

        result = cli("rsqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_rsqrt_2d(self):
        """rsqrt([[4],[25]]"""
        data = [[4.0], [25.0]]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)

        result = cli("rsqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_rsqrt_3d(self):
        """rsqrt on rank-3 tensor"""
        data = [[[4.0], [25.0]], [[9.0], [16.0]]]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)

        result = cli("rsqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_rsqrt_single(self):
        """rsqrt([9])"""
        data = [9.0]
        pt = torch.tensor(data)
        expected = torch.rsqrt(pt)

        result = cli("rsqrt", "--this", str(pt.tolist()))
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Reduction operations  (sum, mean)
# ---------------------------------------------------------------------------

class TestReductionOperations(unittest.TestCase):
    def test_sum_1d(self):
        """sum([1])"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.sum(pt)

        result = cli("sum", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sum_2d(self):
        """sum([[1],[2]])"""
        data = [[1.0], [2.0]]
        pt = torch.tensor(data)  # shape (2, 1), no reshape needed
        expected = torch.sum(pt)

        result = cli("sum", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sum_3d(self):
        """sum on rank-3 tensor with negative values"""
        data = [[[-1.0], [2.0], [3.0]], [[4.0], [-5.0], [6.0]]]
        pt = torch.tensor(data)
        expected = torch.sum(pt)

        result = cli("sum", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_sum_negative(self):
        """sum([-1])"""
        data = [-1.0]
        pt = torch.tensor(data)
        expected = torch.sum(pt)

        result = cli("sum", "--this", str(pt.tolist()))
        verify(self, result, expected)

    def test_mean_1d(self):
        """mean dim=-1 on (1,) -> scalar"""
        data = [2.0]
        pt = torch.tensor(data)
        expected = torch.mean(pt, dim=-1)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "-1")
        verify(self, result, expected)

    def test_mean_2d_dim0(self):
        """mean dim=0 on (2,1) -> (1,)"""
        data = [[1.0], [3.0]]  # shape (2, 1), no reshape needed
        pt = torch.tensor(data).float()
        expected = torch.mean(pt, dim=0)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "0")
        verify(self, result, expected)

    def test_mean_2d_dim1(self):
        """mean dim=1 on (2,1) -> (2,)"""
        data = [[1.0], [3.0]]  # shape (2, 1), no reshape needed
        pt = torch.tensor(data).float()
        expected = torch.mean(pt, dim=1)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "1", "--keepdims", "true")
        verify(self, result, expected)

    def test_mean_3d_dim0(self):
        """mean dim=0 on rank-3 tensor (2,2,2) -> (2,2)"""
        data = list(range(1, 9))  # [1..8], shape (2, 2, 2)
        pt = torch.tensor(data).float().reshape(2, 2, 2)
        expected = torch.mean(pt, dim=0)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "0", "--keepdims", "true")
        verify(self, result, expected)

    def test_mean_3d_dim1(self):
        """mean dim=1 on rank-3 tensor (2,3,2) -> (2,2)"""
        data = list(range(1, 13))  # [1..12], shape (2, 3, 2)
        pt = torch.tensor(data).float().reshape(2, 3, 2)
        expected = torch.mean(pt, dim=1)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "1", "--keepdims", "true")
        verify(self, result, expected)

    def test_mean_keepdims_false(self):
        """mean dim=1 on (2,3) -> (2,)"""
        data = list(range(1, 7))  # [1..6], shape (2, 3)
        pt = torch.tensor(data).float().reshape(2, 3)
        expected = torch.mean(pt, dim=1)

        result = cli("mean", "--this", str(pt.tolist()), "--dim", "1", "--keepdims", "true")
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Advanced operators  (pow, clip)
# ---------------------------------------------------------------------------

class TestAdvancedOperators(unittest.TestCase):
    def test_pow_integer_1d(self):
        """[1]^2"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)

        result = cli("pow", "--this", str(pt.tolist()), "--exponent", "2.0")
        verify(self, result, expected)

    def test_pow_integer_2d(self):
        """[[1],[4]]^2"""
        data = [[1.0], [4.0]]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 2.0)

        result = cli("pow", "--this", str(pt.tolist()), "--exponent", "2.0")
        verify(self, result, expected)

    def test_pow_half(self):
        """[4]^0.5"""
        data = [4.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 0.5)

        result = cli("pow", "--this", str(pt.tolist()), "--exponent", "0.5")
        verify(self, result, expected)

    def test_pow_cube(self):
        """[1]^3"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.pow(pt, 3.0)

        result = cli("pow", "--this", str(pt.tolist()), "--exponent", "3.0")
        verify(self, result, expected)

    def test_clip_1d(self):
        """clip([-5],[-3], min=-1)"""
        data = [-5.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)

        result = cli("clip", "--this", str(pt.tolist()), "--min", "-1.0", "--max", "1.0")
        verify(self, result, expected)

    def test_clip_2d(self):
        """clip([[-5],[-3]], min=-1)"""
        data = [[-5.0], [-3.0]]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)

        result = cli("clip", "--this", str(pt.tolist()), "--min", "-1.0", "--max", "1.0")
        verify(self, result, expected)

    def test_clip_no_op(self):
        """clip([1], min=0) — no clamping"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = torch.clip(pt, min=-1.0, max=1.0)

        result = cli("clip", "--this", str(pt.tolist()), "--min", "-1.0", "--max", "1.0")
        verify(self, result, expected)


# ---------------------------------------------------------------------------
# Type casting  (to)
# ---------------------------------------------------------------------------

class TestTypeCast(unittest.TestCase):
    def test_to_same_type(self):
        """cast to same type (F32->F32) — data unchanged"""
        data = [1.0]
        pt = torch.tensor(data)
        expected = pt.to(torch.float32)

        result = cli("to", "--this", str(pt.tolist()), "--type", "0")  # GGML_TYPE_F32=0
        verify(self, result, expected)


if __name__ == "__main__":
    unittest.main()
