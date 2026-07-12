import unittest
import subprocess
import os
import operator
import torch
from torch.fx.experimental.proxy_tensor import make_fx

class NNTestCase(unittest.TestCase):
    def cli(self, *args: str) -> list:
        cli_bin = os.environ.get('NN_CLI', 'nn-cli')
        result = subprocess.run([cli_bin, *args], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f'nn-cli failed (rc={result.returncode}):\n{result.stderr}')
        return result.stdout.strip().split('\n')

    def traverse(self, fn, *inputs):
        gm = make_fx(fn)(*inputs)

        visited = set()
        graph = []

        def meta(node):
            tm = node.meta.get("tensor_meta", None)

            if tm != None:
                return (tm.dtype, tm.shape)
            
            val = node.meta.get("val", None)

            if val != None:
                return (val.dtype, val.shape)

            return (None, None)

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

        def visit(node, operators = {
            # Arithmetic
            torch.ops.aten.add.Tensor:        "ADD",
            torch.ops.aten.add_.Scalar:       "ADD",
            torch.ops.aten.sub.Tensor:        "SUB",
            torch.ops.aten.mul.Tensor:        "MUL",
            torch.ops.aten.div.Tensor:        "DIV",

            # Unary
            torch.ops.aten.neg.default:       "NEG",
            torch.ops.aten.abs.default:       "ABS",
            torch.ops.aten.sqrt.default:      "SQRT",
            torch.ops.aten.exp.default:       "EXP",
            torch.ops.aten.log.default:       "LOG",
            torch.ops.aten.sin.default:       "SIN",
            torch.ops.aten.cos.default:       "COS",
            torch.ops.aten.tanh.default:      "TANH",

            # Activations
            torch.ops.aten.relu.default:      "RELU",
            torch.ops.aten.silu.default:      "SILU",
            torch.ops.aten.gelu.default:      "GELU",
            torch.ops.aten.sigmoid.default:   "SIGMOID",

            # Tensor ops
            torch.ops.aten.sum.default:       "SUM",
            torch.ops.aten.reshape.default:   "RESHAPE",
            torch.ops.aten.view.default:      "RESHAPE",
            torch.ops.aten.permute.default:   "PERMUTE",
            torch.ops.aten.transpose.int:     "TRANSPOSE",
            torch.ops.aten.unsqueeze.default: "UNSQUEEZE",
            torch.ops.aten.squeeze.dim:       "SQUEEZE",
            torch.ops.aten.cat.default:       "CONCAT",
            torch.ops.aten.t.default:         "TRANSPOSE",

            # Matrix multiplication
            torch.ops.aten.mm.default:        "MUL_MAT",
            torch.ops.aten.bmm.default:       "MUL_MAT",
        }):
            if node in visited:
                return

            dtype, shape = meta(node)

            if not dtype or not shape:
                return

            visited.add(node)

            if node.target is operator.getitem:
                src, idx = node.args

                # native_layer_norm returns (output, mean, rstd)
                if (
                    idx == 0
                    and src.target == torch.ops.aten.native_layer_norm.default
                ):
                    x, normalized_shape, weight, bias, eps = src.args

                    visit(x)
                    emit("NORM", dtype, shape)

                    if weight is not None:
                        wdtype, wshape = meta(weight)
                        visit(weight)
                        emit("MUL", dtype, shape)

                    if bias is not None:
                        bdtype, bshape = meta(bias)
                        visit(bias)
                        emit("ADD", dtype, shape)

            # Handle RMS_NORM: PyTorch outputs raw RMS norm calculation,
            # collapse the graph to just RMS_NORM when such calculation is detected
            elif match := match_rms_norm(node):
                visit(match)
                emit("RMS_NORM", dtype, shape)

            # Handle RSQRT: GGML does not have RSQRT. Expand it to:
            # input -> FILL(1) -> SQRT -> DIV
            elif node.target == torch.ops.aten.rsqrt.default:
                (input,) = node.all_input_nodes

                visit(input)
                emit("FILL", dtype, shape)
                emit("SQRT", dtype, shape)
                emit("DIV", dtype, shape)

            # Handle POW: GGML does not have POW. Expand it to:
            # input -> LOG -> SCALE(exponent) -> EXP
            elif node.target == torch.ops.aten.pow.Tensor_Scalar:
                (input,) = node.all_input_nodes
                exponent = node.args[1]

                visit(input)
                emit("LOG", dtype, shape)
                emit("SCALE", dtype, shape)
                emit("EXP", dtype, shape)

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

            # Handle MEAN: GGML's mean semantics differ from PyTorch.
            # Expand it to:
            # input -> SUM -> CONST(numel) -> DIV
            elif node.target == torch.ops.aten.mean.dim:
                (input,) = node.all_input_nodes

                visit(input)

                emit("SUM", dtype, (1,))

                # Emit divisor = number of reduced elements
                emit("NONE", dtype, (1,))
                emit("FILL", dtype, (1,))

                emit("DIV", dtype, (1,))

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
