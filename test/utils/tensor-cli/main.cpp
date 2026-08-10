#include "../TestCLI.hpp"
#include "ggml/Tensor.hpp"

class TestTensorCLI : public TestCLI {
public:
    TestTensorCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual std::vector<Tensor> compute(Runtime& runtime) {

        if (args_.get(0) == "contiguous") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = self.contiguous();

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "scalar") {
            auto value = args_.get_one<float>("--value");

            auto output = Tensor::scalar(*runtime.context(), value);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "zeros") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::zeros(*runtime.context(), shape);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "ones") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::ones(*runtime.context(), shape);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");

            auto output = Tensor::arange(*runtime.context(), start, stop, step);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        
        if (args_.get(0) == "cat") {
            auto tensors = args_.get_many<Tensor>("--tensor", {runtime});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::cat(tensors, dim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "stack") {
            auto tensors = args_.get_many<Tensor>("--tensor", {runtime});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::stack(tensors, dim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "reshape") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.reshape(shape);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "permute") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto order = args_.get_one<Tensor::Shape>("--order");

            auto output = self.permute(order);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "squeeze") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.squeeze(dim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "unsqueeze") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.unsqueeze(dim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "flatten") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            auto output = self.flatten(start_dim, end_dim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "unflatten") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.unflatten(dim, shape);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "narrow") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            auto output = self.narrow(dim, start, length);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "expand") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            auto output = self.expand(new_shape);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "repeat") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto repeats = args_.get_one<Tensor::Shape>("--repeats");

            auto output = self.repeat(repeats);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "chunk") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto n = args_.get_one<int>("--n");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.chunk(n, dim);

            Graph graph(runtime, std::move(output));
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "split") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto split_size = args_.get_one<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split(split_size, dim);

            Graph graph(runtime, std::move(output));
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "split_with_sizes") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto split_sizes = args_.get_many<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split_with_sizes(split_sizes, dim);

            Graph graph(runtime, std::move(output));
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "to") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto type = (ggml_type)args_.get_one<int>("--type");

            auto output = self.to(type);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "neg") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            
            auto output = -self;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "add") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs + rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "sub") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs - rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "mul") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs * rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "div") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs / rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "add_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs + rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "sub_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs - rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "mul_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs * rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "div_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {runtime});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs / rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "scalar_add") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs + rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "scalar_sub") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs - rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "scalar_mul") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs * rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "scalar_div") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {runtime});

            auto output = lhs / rhs;

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "pow_scalar") {
            auto base = args_.get_one<Tensor>("--base", {runtime});
            auto exponent = args_.get_one<float>("--exponent");

            auto output = pow(base, exponent);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "scalar_pow") {
            auto base = args_.get_one<float>("--base");
            auto exponent = args_.get_one<Tensor>("--exponent", {runtime});

            auto output = pow(base, exponent);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        
        if (args_.get(0) == "clamp") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            auto output = self.clamp(min, max);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "sum") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.sum(dim, keepdim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "mean") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.mean(dim, keepdim);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "index") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto index = args_.get_one<size_t>("--index");

            auto output = self[index];

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "slice") {
            auto self = args_.get_one<Tensor>("--this", {runtime});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            auto output = self[slice];

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }


        if (args_.get(0) == "abs") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = abs(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "sqrt") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = sqrt(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "exp") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = exp(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "log") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = log(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "sin") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = sin(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "cos") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = cos(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "rsqrt") {
            auto self = args_.get_one<Tensor>("--this", {runtime});

            auto output = rsqrt(self);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestTensorCLI cli(argc, argv);
    return cli.main();
}
