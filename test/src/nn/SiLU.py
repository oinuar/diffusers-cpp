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
        torch.ops.aten.add.Tensor: "ADD",
        torch.ops.aten.add_.Tensor: "ADD",
        torch.ops.aten.sub.Tensor: "SUB",
        torch.ops.aten.mul.Tensor: "MUL",
        torch.ops.aten.mul_.Tensor: "MUL",
        torch.ops.aten.div.Tensor: "DIV",
        torch.ops.aten.mm.default: "MUL_MAT",
        torch.ops.aten.matmul.default: "MUL_MAT",
        torch.ops.aten.relu.default: "RELU",
        torch.ops.aten.silu.default: "SILU",
        torch.ops.aten.tanh.default: "TANH",
        torch.ops.aten.sqrt.default: "SQRT",
        torch.ops.aten.exp.default: "EXP",
        torch.ops.aten.log.default: "LOG",
        torch.ops.aten.sin.default: "SIN",
        torch.ops.aten.cos.default: "COS",
        torch.ops.aten.mm.default: "MUL_MAT",
        torch.ops.aten.t.default: "TRANSPOSE"
    }):
        if node in visited:
            return

        dtype, shape = meta(node)

        if not dtype or not shape:
            return

        visited.add(node)

        # Handle ADDM: GGML does not have ADDM operator. Expand it to:
        # weight -> CONT -> input -> MUL_MAT -> bias -> ADD
        if node.target == torch.ops.aten.addmm.default:
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


class TestNNSiLU(unittest.TestCase):
    def test_1d(self):
        model = nn.SiLU()
        expected = traverse(model, torch.randn(2))
        actual = cli('silu', '--input', '(2)')
        self.assertEqual(actual, expected)
