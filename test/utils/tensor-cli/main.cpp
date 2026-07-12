#include "ggml/Compute.hpp"
#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "../ArgumentParser.hpp"

class TensorCLI : public Compute {
public:
    TensorCLI(int argc, char** argv) : args_(argc, argv) {}

    Plan build(Context& ctx) {

        if (args_.get(0) == "contiguous") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.contiguous();
        }


        if (args_.get(0) == "scalar") {
            auto value = args_.get_one<float>("--value");

            return Tensor::scalar(*ctx, value);
        }

        if (args_.get(0) == "zeros") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return Tensor::zeros(*ctx, shape);
        }

        if (args_.get(0) == "ones") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return Tensor::ones(*ctx, shape);
        }

        if (args_.get(0) == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");

            return Tensor::arange(*ctx, start, stop, step);
        }

        
        if (args_.get(0) == "cat") {
            auto tensors = args_.get_many<Tensor>("--tensor", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return Tensor::cat(tensors, dim);
        }


        if (args_.get(0) == "reshape") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return self.reshape(shape);
        }

        if (args_.get(0) == "permute") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto order = args_.get_one<Tensor::Shape>("--order");

            return self.permute(order);
        }

        if (args_.get(0) == "squeeze") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return self.squeeze(dim);
        }

        if (args_.get(0) == "unsqueeze") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return self.unsqueeze(dim);
        }

        if (args_.get(0) == "flatten") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            return self.flatten(start_dim, end_dim);
        }

        if (args_.get(0) == "unflatten") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return self.unflatten(dim, shape);
        }

        if (args_.get(0) == "narrow") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            return self.narrow(dim, start, length);
        }

        if (args_.get(0) == "expand") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            return self.expand(new_shape);
        }

        if (args_.get(0) == "chunk") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto n = args_.get_one<int>("--n");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            return self.chunk(n, dim);
        }

        if (args_.get(0) == "split") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto split_size = args_.get_one<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            return self.split(split_size, dim);
        }

        if (args_.get(0) == "split_with_sizes") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto split_sizes = args_.get_many<int64_t>("--split_size");
            auto dim = args_.get_optional<int>("--dim").value_or(0);

            return self.split_with_sizes(split_sizes, dim);
        }

        if (args_.get(0) == "to") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto type = (ggml_type)args_.get_one<int>("--type");

            return self.to(type);
        }


        if (args_.get(0) == "neg") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            
            return -self;
        }

        if (args_.get(0) == "add") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs + rhs;
        }

        if (args_.get(0) == "sub") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs - rhs;
        }

        if (args_.get(0) == "mul") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs * rhs;
        }

        if (args_.get(0) == "div") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs / rhs;
        }

        if (args_.get(0) == "add_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs + rhs;
        }

        if (args_.get(0) == "sub_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs - rhs;
        }

        if (args_.get(0) == "mul_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs * rhs;
        }

        if (args_.get(0) == "div_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs / rhs;
        }


        if (args_.get(0) == "scalar_add") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs + rhs;
        }

        if (args_.get(0) == "scalar_sub") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs - rhs;
        }

        if (args_.get(0) == "scalar_mul") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs * rhs;
        }

        if (args_.get(0) == "scalar_div") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs / rhs;
        }

        if (args_.get(0) == "pow") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto exponent = args_.get_one<float>("--exponent");

            return self.pow(exponent);
        }

        
        if (args_.get(0) == "clip") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            return self.clip(min, max);
        }

        if (args_.get(0) == "sum") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.sum();
        }

        if (args_.get(0) == "mean") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.mean();
        }


        if (args_.get(0) == "index") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto index = args_.get_one<size_t>("--index");

            return self[index];
        }

        if (args_.get(0) == "slice") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            return self[slice];
        }


        if (args_.get(0) == "abs") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return abs(self);
        }

        if (args_.get(0) == "sqrt") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return sqrt(self);
        }

        if (args_.get(0) == "exp") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return exp(self);
        }

        if (args_.get(0) == "log") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return log(self);
        }

        if (args_.get(0) == "sin") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return sin(self);
        }

        if (args_.get(0) == "cos") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return cos(self);
        }

        if (args_.get(0) == "rsqrt") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return rsqrt(self);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

    void compute(Graph& graph) {
        Computation computation(graph);

        for (auto& [tensor, data] : inputs_)
            computation.load(tensor, reinterpret_cast<std::byte*>(data.data()));

        computation.execute();

        auto result = computation.read<float>();

        for (auto& [shape, data] : result) {
            std::cerr << "output shape: " << shape.to_string() << std::endl;
            print_tensor(data, shape);
        }
    }

private:
    ArgumentParser args_;
    std::vector<std::pair<Tensor, std::vector<float>>> inputs_;

    template <typename T>
    void print_tensor(const std::vector<T>& data, const Tensor::Shape& shape) {
        size_t expected = 1;
        for (auto i = 0; i < shape.rank(); ++i)
            expected *= shape[i];

        if (expected != data.size())
            throw std::runtime_error("tensor data size does not match shape");

        size_t index = 0;
        print_tensor_recursively(data, shape, 0, index);
        std::cout << std::endl;
    }

    template <typename T>
    void print_tensor_recursively(const std::vector<T>& data, const Tensor::Shape& shape, size_t dim, size_t& index) {
        if (shape.rank() == 0 && dim == 0) {
            std::cout << data[index++];
            return;
        }

        std::cout << "[";

        if (dim == shape.rank() - 1)
        {
            // Last dimension: print elements
            for (auto i = 0; i < shape[dim]; ++i)
            {
                std::cout << data[index++];
                if (i + 1 != shape[dim])
                    std::cout << ", ";
            }
        }
        else
        {
            // Print nested arrays
            for (auto i = 0; i < shape[dim]; ++i)
            {
                print_tensor_recursively(data, shape, dim + 1, index);
                if (i + 1 != shape[dim])
                    std::cout << ", ";
            }
        }

        std::cout << "]";
    }
};

int main(int argc, char** argv) {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Scheduler scheduler({*cpu});

    TensorCLI tensor_cli(argc, argv);

    auto graph = scheduler.plan(tensor_cli);
    tensor_cli.compute(graph);

    return EXIT_SUCCESS;
}
