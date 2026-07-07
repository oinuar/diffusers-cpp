#include "ggml/Compute.hpp"
#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "../../ArgumentParser.hpp"
#include "./TensorParser.hpp"
#include "./ShapeParser.hpp"
#include "./SliceParser.hpp"

class TensorCLI : public Compute {
public:
    TensorCLI(int argc, char** argv) : args_(argc, argv) {}

    Tensor build(Context& ctx) {

        if (args_.command() == "contiguous") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.contiguous();
        }


        if (args_.command() == "scalar") {
            auto value = args_.get_one<float>("--value");

            return Tensor::scalar(*ctx, value);
        }

        if (args_.command() == "zeros") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return Tensor::zeros(*ctx, shape);
        }

        if (args_.command() == "ones") {
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return Tensor::ones(*ctx, shape);
        }

        if (args_.command() == "arange") {
            auto start = args_.get_one<float>("--start");
            auto stop = args_.get_one<float>("--stop");
            auto step = args_.get_one<float>("--step");

            return Tensor::arange(*ctx, start, stop, step);
        }

        
        if (args_.command() == "cat") {
            auto tensors = args_.get_many<Tensor>("--tensor", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return Tensor::cat(tensors, dim);
        }


        if (args_.command() == "reshape") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return self.reshape(shape);
        }

        if (args_.command() == "permute") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto order = args_.get_one<Tensor::Shape>("--order");

            return self.permute(order);
        }

        if (args_.command() == "squeeze") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return self.squeeze(dim);
        }

        if (args_.command() == "unsqueeze") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");

            return self.unsqueeze(dim);
        }

        if (args_.command() == "flatten") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto start_dim = args_.get_optional<int>("--start_dim").value_or(0);
            auto end_dim = args_.get_optional<int>("--end_dim").value_or(-1);

            return self.flatten(start_dim, end_dim);
        }

        if (args_.command() == "unflatten") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int64_t>("--dim");
            auto shape = args_.get_one<Tensor::Shape>("--shape");

            return self.unflatten(dim, shape);
        }

        if (args_.command() == "narrow") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto dim = args_.get_one<int>("--dim");
            auto start = args_.get_one<int64_t>("--start");
            auto length = args_.get_one<int64_t>("--length");

            return self.narrow(dim, start, length);
        }

        if (args_.command() == "expand") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto new_shape = args_.get_one<Tensor::Shape>("--new-shape");

            return self.expand(new_shape);
        }

        if (args_.command() == "chunk") {
            throw std::runtime_error("not implemented yet");
        }

        if (args_.command() == "split") {
            throw std::runtime_error("not implemented yet");
        }

        if (args_.command() == "split_with_sizes") {
            throw std::runtime_error("not implemented yet");
        }

        if (args_.command() == "to") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto type = (ggml_type)args_.get_one<int>("--type");

            return self.to(type);
        }


        if (args_.command() == "neg") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            
            return -self;
        }

        if (args_.command() == "add") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs + rhs;
        }

        if (args_.command() == "sub") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs - rhs;
        }

        if (args_.command() == "mul") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs * rhs;
        }

        if (args_.command() == "div") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs / rhs;
        }

        if (args_.command() == "add_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs + rhs;
        }

        if (args_.command() == "sub_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs - rhs;
        }

        if (args_.command() == "mul_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs * rhs;
        }

        if (args_.command() == "div_scalar") {
            auto lhs = args_.get_one<Tensor>("--lhs", {ctx, inputs_});
            auto rhs = args_.get_one<float>("--rhs");

            return lhs / rhs;
        }


        if (args_.command() == "scalar_add") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs + rhs;
        }

        if (args_.command() == "scalar_sub") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs - rhs;
        }

        if (args_.command() == "scalar_mul") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs * rhs;
        }

        if (args_.command() == "scalar_div") {
            auto lhs = args_.get_one<float>("--lhs");
            auto rhs = args_.get_one<Tensor>("--rhs", {ctx, inputs_});

            return lhs / rhs;
        }

        if (args_.command() == "pow") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto exponent = args_.get_one<float>("--exponent");

            return self.pow(exponent);
        }

        
        if (args_.command() == "clip") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto min = args_.get_one<float>("--min");
            auto max = args_.get_one<float>("--max");

            return self.clip(min, max);
        }

        if (args_.command() == "sum") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.sum();
        }

        if (args_.command() == "mean") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return self.mean();
        }


        if (args_.command() == "index") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto index = args_.get_one<size_t>("--index");

            return self[index];
        }

        if (args_.command() == "slice") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});
            auto slice = args_.get_one<std::vector<Tensor::Slice>>("--slice");

            return self[slice];
        }


        if (args_.command() == "abs") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return abs(self);
        }

        if (args_.command() == "sqrt") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return sqrt(self);
        }

        if (args_.command() == "exp") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return exp(self);
        }

        if (args_.command() == "log") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return log(self);
        }

        if (args_.command() == "sin") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return sin(self);
        }

        if (args_.command() == "cos") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return cos(self);
        }

        if (args_.command() == "rsqrt") {
            auto self = args_.get_one<Tensor>("--this", {ctx, inputs_});

            return rsqrt(self);
        }

        throw std::runtime_error("Uknown command: " + args_.command());
    }

    void compute(Graph& graph) {
        Computation computation(graph);

        for (auto& [tensor, data] : inputs_)
            computation.load(tensor, reinterpret_cast<std::byte*>(data.data()));

        computation.execute();

        auto [shape, data] = computation.read<float>();

        std::cerr << "output shape: " << shape.to_string() << std::endl;

        print_tensor(data, shape);
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
