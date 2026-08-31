#include "../TestCLI.hpp"
#include "ggml/Tensor.hpp"

class TestTensorCLI : public TestCLI {
public:
    TestTensorCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual std::vector<Tensor> compute(Scheduler& scheduler, Context& context, Allocator* allocator) {

        if (args_.get(0) == "contiguous") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = self.contiguous();

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "scalar") {
            auto value = args_.get_one<float>("--value");

            auto output = Tensor::scalar(*context, value);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "zeros") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::zeros(*context, shape);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "ones") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::ones(*context, shape);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");

            auto output = context.arange(start, stop, step);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        
        if (args_.get(0) == "cat") {
            auto tensors = args_.get_many<Tensor>("--tensor", {context});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::cat(tensors, dim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "stack") {
            auto tensors = args_.get_many<Tensor>("--tensor", {context});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::stack(tensors, dim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }


        if (args_.get(0) == "reshape") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.reshape(shape);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "permute") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto order = args_.get_one<Tensor::Shape>("--order");

            auto output = self.permute(order);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "squeeze") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.squeeze(dim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "unsqueeze") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.unsqueeze(dim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "flatten") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            auto output = self.flatten(start_dim, end_dim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "unflatten") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.unflatten(dim, shape);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "narrow") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            auto output = self.narrow(dim, start, length);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "expand") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            auto output = self.expand(new_shape);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "repeat") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto repeats = args_.get_one<Tensor::Shape>("--repeats");

            auto output = self.repeat(repeats);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "chunk") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto n = args_.get_one<int>("--n");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.chunk(n, dim);

            Graph graph(scheduler, context, std::move(output));

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "split") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto split_size = args_.get_one<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split(split_size, dim);

            Graph graph(scheduler, context, std::move(output));

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "split_with_sizes") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto split_sizes = args_.get_many<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split_with_sizes(split_sizes, dim);

            Graph graph(scheduler, context, std::move(output));

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "to") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto type = (ggml_type)args_.get_one<int>("--type");

            auto output = self.to(type);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }


        if (args_.get(0) == "neg") {
            auto self = args_.get_one<Tensor>("--this", {context});
            
            auto output = -self;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "add") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs + rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "sub") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs - rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "mul") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs * rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "div") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs / rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "add_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs + rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "sub_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs - rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "mul_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs * rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "div_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {context});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs / rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }


        if (args_.get(0) == "scalar_add") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs + rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "scalar_sub") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs - rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "scalar_mul") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs * rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "scalar_div") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {context});

            auto output = lhs / rhs;

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "pow_scalar") {
            auto base = args_.get_one<Tensor>("--base", {context});
            auto exponent = args_.get_one<float>("--exponent");

            auto output = pow(base, exponent);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "scalar_pow") {
            auto base = args_.get_one<float>("--base");
            auto exponent = args_.get_one<Tensor>("--exponent", {context});

            auto output = pow(base, exponent);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        
        if (args_.get(0) == "clamp") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            auto output = self.clamp(min, max);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "sum") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.sum(dim, keepdim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "mean") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.mean(dim, keepdim);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }


        if (args_.get(0) == "index") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto index = args_.get_one<size_t>("--index");

            auto output = self[index];

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "slice") {
            auto self = args_.get_one<Tensor>("--this", {context});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            auto output = self[slice];

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }


        if (args_.get(0) == "abs") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = abs(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "sqrt") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = sqrt(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "exp") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = exp(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "log") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = log(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "sin") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = sin(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "cos") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = cos(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        if (args_.get(0) == "rsqrt") {
            auto self = args_.get_one<Tensor>("--this", {context});

            auto output = rsqrt(self);

            Graph graph(scheduler, context, {output});

            if (allocator != nullptr)
                allocator->allocate();

            Computation computation(graph);
            return computation().results();
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestTensorCLI cli(argc, argv);
    return cli.main();
}
