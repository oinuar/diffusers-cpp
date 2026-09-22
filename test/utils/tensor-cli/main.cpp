#include "../TestCLI.hpp"
#include "ggml/Tensor.hpp"

class TestTensorCLI : public TestCLI {
public:
    TestTensorCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Computation<std::vector<Tensor>> compute(Context& context) {

        if (args_.get(0) == "contiguous") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.contiguous();
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "scalar") {
            Computation<void> computation({&context});
            auto value = args_.get_one<float>("--value");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return Tensor::scalar(value);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "zeros") {
            Computation<void> computation({&context});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return Tensor::zeros(shape);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "ones") {
            Computation<void> computation({&context});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return Tensor::ones(shape);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");
            Computation<void> computation({&context});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return scope.context().arange(start, stop, step);
            });

            return Computation<Tensor>::all(result);
        }

        
        if (args_.get(0) == "cat") {
            Computation<void> computation({&context});
            auto tensors = args_.get_many<Tensor>("--tensor", {computation.desc()->context()});
            auto dim = args_.get_one<int>("--dim");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return Tensor::cat(tensors, dim);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "stack") {
            Computation<void> computation({&context});
            auto tensors = args_.get_many<Tensor>("--tensor", {computation.desc()->context()});
            auto dim = args_.get_one<int>("--dim");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return Tensor::stack(tensors, dim);
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "reshape") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.reshape(shape);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "permute") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto order = args_.get_one<Tensor::Shape>("--order");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.permute(order);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "squeeze") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_one<int>("--dim");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.squeeze(dim);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "unsqueeze") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_one<int>("--dim");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.unsqueeze(dim);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "flatten") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.flatten(start_dim, end_dim);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "unflatten") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.unflatten(dim, shape);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "narrow") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.narrow(dim, start, length);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "expand") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.expand(new_shape);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "repeat") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto repeats = args_.get_one<Tensor::Shape>("--repeats");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.repeat(repeats);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "chunk") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto n = args_.get_one<int>("--n");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                return self.chunk(n, dim);
            });

            return result;
        }

        if (args_.get(0) == "split") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto split_size = args_.get_one<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                return self.split(split_size, dim);
            });

            return result;
        }

        if (args_.get(0) == "split_with_sizes") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto split_sizes = args_.get_many<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto result = computation.scope([&](Scope scope) -> std::vector<Tensor> {
                return self.split_with_sizes(split_sizes, dim);
            });

            return result;
        }

        
        if (args_.get(0) == "neg") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return -self;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "add") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs + rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "sub") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs - rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "mul") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs * rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "div") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs / rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "add_scalar") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs + rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "sub_scalar") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs - rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "mul_scalar") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs * rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "div_scalar") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<Tensor>("--lhs", {computation.desc()->context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs / rhs;
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "scalar_add") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs + rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "scalar_sub") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs - rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "scalar_mul") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs * rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "scalar_div") {
            Computation<void> computation({&context});
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return lhs / rhs;
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "pow_scalar") {
            Computation<void> computation({&context});
            auto base = args_.get_one<Tensor>("--base", {computation.desc()->context()});
            auto exponent = args_.get_one<float>("--exponent");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return pow(base, exponent);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "scalar_pow") {
            Computation<void> computation({&context});
            auto base = args_.get_one<float>("--base");
            auto exponent = args_.get_one<Tensor>("--exponent", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return pow(base, exponent);
            });

            return Computation<Tensor>::all(result);
        }

        
        if (args_.get(0) == "clamp") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.clamp(min, max);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "sum") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.sum(dim, keepdim);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "mean") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self.mean(dim, keepdim);
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "index") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto index = args_.get_one<size_t>("--index");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self[index];
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "slice") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return self[slice];
            });

            return Computation<Tensor>::all(result);
        }


        if (args_.get(0) == "abs") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return abs(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "sqrt") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return sqrt(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "exp") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return exp(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "log") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return log(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "sin") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return sin(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "cos") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return cos(self);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "rsqrt") {
            Computation<void> computation({&context});
            auto self = args_.get_one<Tensor>("--this", {computation.desc()->context()});

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return rsqrt(self);
            });

            return Computation<Tensor>::all(result);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestTensorCLI cli(argc, argv);
    return cli.main();
}
