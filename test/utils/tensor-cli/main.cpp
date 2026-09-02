#include "../TestCLI.hpp"
#include "ggml/Tensor.hpp"

class TestTensorCLI : public TestCLI {
public:
    TestTensorCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual std::vector<Tensor> compute(Scheduler& scheduler, Context& context, Allocator& allocator, std::optional<Context>& local_context, std::optional<DeviceAllocator>& local_allocator) {

        if (args_.get(0) == "contiguous") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = self.contiguous();
        
            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "scalar") {
            Scope scope(local_context ? *local_context : context);
            auto value = args_.get_one<float>("--value");

            auto output = Tensor::scalar(value);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "zeros") {
            Scope scope(local_context ? *local_context : context);
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::zeros(shape);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "ones") {
            Scope scope(local_context ? *local_context : context);
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = Tensor::ones(shape);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");
            Scope scope(local_context ? *local_context : context);

            auto output = scope.context().arange(start, stop, step);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        
        if (args_.get(0) == "cat") {
            Scope scope(local_context ? *local_context : context);
            auto tensors = args_.get_many<Tensor>("--tensor", {scope.context()});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::cat(tensors, dim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "stack") {
            Scope scope(local_context ? *local_context : context);
            auto tensors = args_.get_many<Tensor>("--tensor", {scope.context()});
            auto dim = args_.get_one<int>("--dim");

            auto output = Tensor::stack(tensors, dim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }


        if (args_.get(0) == "reshape") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.reshape(shape);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "permute") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto order = args_.get_one<Tensor::Shape>("--order");

            auto output = self.permute(order);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "squeeze") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.squeeze(dim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "unsqueeze") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_one<int>("--dim");

            auto output = self.unsqueeze(dim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "flatten") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            auto output = self.flatten(start_dim, end_dim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "unflatten") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            auto output = self.unflatten(dim, shape);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "narrow") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            auto output = self.narrow(dim, start, length);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "expand") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            auto output = self.expand(new_shape);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "repeat") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto repeats = args_.get_one<Tensor::Shape>("--repeats");

            auto output = self.repeat(repeats);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "chunk") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto n = args_.get_one<int>("--n");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.chunk(n, dim);

            Graph graph(scheduler, context, std::move(output));

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "split") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto split_size = args_.get_one<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split(split_size, dim);

            Graph graph(scheduler, context, std::move(output));

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "split_with_sizes") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto split_sizes = args_.get_many<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            auto output = self.split_with_sizes(split_sizes, dim);

            Graph graph(scheduler, context, std::move(output));

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "to") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto type = (ggml_type)args_.get_one<int>("--type");

            auto output = self.to(type);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }


        if (args_.get(0) == "neg") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = -self;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "add") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs + rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "sub") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs - rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "mul") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs * rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "div") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs / rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "add_scalar") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs + rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "sub_scalar") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs - rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "mul_scalar") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs * rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "div_scalar") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<Tensor>("--lhs", {scope.context()});
            auto rhs = args_.get_one<float>("--rhs");

            auto output = lhs / rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }


        if (args_.get(0) == "scalar_add") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs + rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "scalar_sub") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs - rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "scalar_mul") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs * rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "scalar_div") {
            Scope scope(local_context ? *local_context : context);
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {scope.context()});

            auto output = lhs / rhs;

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "pow_scalar") {
            Scope scope(local_context ? *local_context : context);
            auto base = args_.get_one<Tensor>("--base", {scope.context()});
            auto exponent = args_.get_one<float>("--exponent");

            auto output = pow(base, exponent);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "scalar_pow") {
            Scope scope(local_context ? *local_context : context);
            auto base = args_.get_one<float>("--base");
            auto exponent = args_.get_one<Tensor>("--exponent", {scope.context()});

            auto output = pow(base, exponent);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        
        if (args_.get(0) == "clamp") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            auto output = self.clamp(min, max);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "sum") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.sum(dim, keepdim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "mean") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto dim = args_.get_optional<int64_t>("--dim").value_or(-1);
            auto keepdim = args_.get_optional<bool>("--keepdim").value_or(false);

            auto output = self.mean(dim, keepdim);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }


        if (args_.get(0) == "index") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto index = args_.get_one<size_t>("--index");

            auto output = self[index];

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "slice") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            auto output = self[slice];

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }


        if (args_.get(0) == "abs") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = abs(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "sqrt") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = sqrt(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "exp") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = exp(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "log") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = log(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "sin") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = sin(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "cos") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = cos(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "rsqrt") {
            Scope scope(local_context ? *local_context : context);
            auto self = args_.get_one<Tensor>("--this", {scope.context()});

            auto output = rsqrt(self);

            Graph graph(scheduler, scope.context(), {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, &scope.context()});
            return computation().results();
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestTensorCLI cli(argc, argv);
    return cli.main();
}
