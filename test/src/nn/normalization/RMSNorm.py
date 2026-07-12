import torch
from torch.fx.experimental.proxy_tensor import make_fx
import torch.nn as nn
import subprocess
import unittest
import os

def cli(*args: str) -> list:
    cli_bin = os.environ.get('NN_CLI', 'nn-cli')
    result = subprocess.run([cli_bin, *args], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'nn-cli failed (rc={result.returncode}):\n{result.stderr}')
    return result.stdout.strip().split('\n')

def meta(node):
    tm = node.meta.get("tensor_meta", None)

    if tm != None:
        return (tm.dtype, tm.shape)
    
    val = node.meta.get("val", None)

    if val != None:
        return (val.dtype, val.shape)

    return (None, None)

def match_rms_norm(node):
    # Match:
    # x ----\
    #        MUL
    # rsqrt -/
    if node.target != torch.ops.aten.mul.Tensor:
        return None

    lhs, rhs = node.all_input_nodes

    if lhs.target == torch.ops.aten.rsqrt.default:
        rsqrt, x = lhs, rhs
    elif rhs.target == torch.ops.aten.rsqrt.default:
        rsqrt, x = rhs, lhs
    else:
        return None

    add = rsqrt.all_input_nodes[0]
    if add.target != torch.ops.aten.add_.Scalar:
        return None

    mean = add.all_input_nodes[0]
    if mean.target != torch.ops.aten.mean.dim:
        return None

    pow = mean.all_input_nodes[0]
    if pow.target != torch.ops.aten.pow.Tensor_Scalar:
        return None

    if pow.all_input_nodes[0] is not x:
        return None

    # Verify exponent == 2
    if pow.args[1] != 2:
        return None

    # Extract eps
    eps = add.args[1]

    return x

def match_layer_norm(node):
    if node.target != torch.ops.aten.mul.Tensor:
        return None

    lhs, rhs = node.all_input_nodes

    if lhs.target == torch.ops.aten.rsqrt.default:
        rsqrt, sub = lhs, rhs
    elif rhs.target == torch.ops.aten.rsqrt.default:
        rsqrt, sub = rhs, lhs
    else:
        return None

    if sub.target != torch.ops.aten.sub.Tensor:
        return None

    x, mean1 = sub.all_input_nodes

    if mean1.target != torch.ops.aten.mean.dim:
        return None

    if mean1.all_input_nodes[0] is not x:
        return None

    add = rsqrt.all_input_nodes[0]
    if add.target != torch.ops.aten.add_.Scalar:
        return None

    mean2 = add.all_input_nodes[0]
    if mean2.target != torch.ops.aten.mean.dim:
        return None

    pow = mean2.all_input_nodes[0]
    if pow.target != torch.ops.aten.pow.Tensor_Scalar:
        return None

    if pow.args[1] != 2:
        return None

    if pow.all_input_nodes[0] is not sub:
        return None

    return x

def traverse(fn, *inputs):
    gm = make_fx(fn)(*inputs)

    visited = set()
    graph = []

    def emit(op, dtype, shape, types={
        torch.float16: "f16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.bool: "bool",
    }):
        # Collapse leading singleton dimensions like GGML
        i = 0
        while i < len(shape) - 1 and shape[i] == 1:
            i += 1
        shape = shape[i:]

        graph.append(f"{op} {types[dtype]} ({', '.join(map(str, shape))})")

    def visit(node, operators={
        torch.ops.aten.add_.Scalar: "ADD",
        torch.ops.aten.mul.Tensor: "MUL",
        torch.ops.aten.rsqrt.default: "RSQRT",
        torch.ops.aten.pow.Tensor_Scalar: "POW",
        torch.ops.aten.mean.dim: "MEAN",
        torch.ops.aten.rsqrt.default: "RSQRT" 
    }):
        if node in visited:
            return

        dtype, shape = meta(node)

        if not dtype or not shape:
            return

        visited.add(node)

        # Handle RMS_NORM: PyTorch outputs raw RMS norm calculation,
        # collapse the graph to just RMS_NORM when such calculation is detected
        if match := match_rms_norm(node):
            visit(match)
            emit("RMS_NORM", dtype, shape)

        # Handle NORM: PyTorch outputs raw layer norm calucation,
        # collapse the graph to just NORM when such calculation is detected
        elif match := match_layer_norm(node):
            visit(match)
            emit("NORM", dtype, shape)

        # Handle ADDM: GGML does not have ADDM operator. Expand it to:
        # weight -> CONT -> input -> MUL_MAT -> bias -> ADD
        elif node.target == torch.ops.aten.addmm.default:
            # Inputs are: bias, input, weight
            bias, input, weight = node.all_input_nodes
            wdtype, wshape = meta(weight)

            visit(weight)
            emit("CONT", wdtype, wshape)
            visit(input)
            emit("MUL_MAT", dtype, shape)
            visit(bias)
            emit("ADD", dtype, shape)
        
        # Otherwise, traverse DFS
        else:
            for inp in node.all_input_nodes:
                visit(inp)

            emit("NONE" if node.op in ("placeholder", "get_attr") else operators[node.target], dtype, shape)

    # Start from the graph output(s)
    output_node = next(n for n in gm.graph.nodes if n.op == "output")

    for inp in output_node.all_input_nodes:
        visit(inp)

    return graph


class TestNNNormalizationRMSNorm(unittest.TestCase):
    def test_dim3(self):
        model = nn.modules.normalization.RMSNorm(3)
        expected = traverse(model, torch.randn(2, 2, 3))
        actual = cli('rmsnorm', '--dim', '3', '--input', '(2, 2, 3)')
        self.assertEqual(actual, expected)
