#include <functional>
#include "GGMLCompute.hpp"
#include "GGMLGraph.hpp"
#include "GGMLComputation.hpp"
#include "GGMLContext.hpp"
#include "GGMLBackend.hpp"
#include "GGMLScheduler.hpp"
#include "../ArgumentParser.hpp"
#include "../TensorParser.hpp"
#include "../ShapeParser.hpp"
#include "../SliceParser.hpp"

class TensorCLI : public GGMLCompute {
public:
    TensorCLI(int argc, char** argv) : args_(argc, argv) {}

    Tensor build(GGMLContext& ctx) {

        if (args_.command() == "contiguous") {
            auto self = get_tensor(ctx, "--this");

            return self.contiguous();
        }


        if (args_.command() == "scalar") {
            auto value = get_value<float>("--value");

            return Tensor::scalar(*ctx, value);
        }

        if (args_.command() == "zeros") {
            auto shape = get_shape("--shape");

            return Tensor::zeros(*ctx, shape);
        }

        if (args_.command() == "ones") {
            auto shape = get_shape("--shape");

            return Tensor::ones(*ctx, shape);
        }

        if (args_.command() == "arange") {
            auto start = get_value<float>("--start");
            auto stop = get_value<float>("--stop");
            auto step = get_value<float>("--step");

            return Tensor::arange(*ctx, start, stop, step);
        }

        
        if (args_.command() == "cat") {
            /*auto tensors = args_.get<std::vector<Tensor>>("--tensor");

            return Tensor::cat(tensors, get_value<int>("--dim"));*/
            throw std::runtime_error("not implemented yet");
        }


        if (args_.command() == "reshape") {
            auto self = get_tensor(ctx, "--this");
            auto shape = get_shape("--shape");

            return self.reshape(shape);
        }

        if (args_.command() == "permute") {
            auto self = get_tensor(ctx, "--this");
            auto order = get_shape("--order");

            return self.permute(order);
        }

        if (args_.command() == "squeeze") {
            auto self = get_tensor(ctx, "--this");
            auto dim = get_value<int>("--dim");

            return self.squeeze(dim);
        }

        if (args_.command() == "unsqueeze") {
            auto self = get_tensor(ctx, "--this");
            auto dim = get_value<int>("--dim");

            return self.unsqueeze(dim);
        }

        if (args_.command() == "flatten") {
            auto self = get_tensor(ctx, "--this");
            auto start_dim = get_value<int>("--start_dim", 0);
            auto end_dim = get_value<int>("--end_dim", -1);

            return self.flatten(start_dim, end_dim);
        }

        if (args_.command() == "unflatten") {
            auto self = get_tensor(ctx, "--this");
            auto dim = get_value<int64_t>("--dim");
            auto shape = get_shape("--shape");

            return self.unflatten(dim, shape);
        }

        if (args_.command() == "narrow") {
            auto self = get_tensor(ctx, "--this");
            auto dim = get_value<int>("--dim");
            auto start = get_value<int64_t>("--start");
            auto length = get_value<int64_t>("--length");

            return self.narrow(dim, start, length);
        }

        if (args_.command() == "expand") {
            auto self = get_tensor(ctx, "--this");
            auto new_shape = get_shape("--new-shape");

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
            auto self = get_tensor(ctx, "--this");
            auto type = (ggml_type)get_value<int>("--type");

            return self.to(type);
        }


        if (args_.command() == "neg") {
            auto self = get_tensor(ctx, "--this");
            
            return -self;
        }

        if (args_.command() == "add") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs + rhs;
        }

        if (args_.command() == "sub") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs - rhs;
        }

        if (args_.command() == "mul") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs * rhs;
        }

        if (args_.command() == "div") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs / rhs;
        }

        if (args_.command() == "add_scalar") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_value<float>("--rhs");

            return lhs + rhs;
        }

        if (args_.command() == "sub_scalar") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_value<float>("--rhs");

            return lhs - rhs;
        }

        if (args_.command() == "mul_scalar") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_value<float>("--rhs");

            return lhs * rhs;
        }

        if (args_.command() == "div_scalar") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_value<float>("--rhs");

            return lhs / rhs;
        }


        if (args_.command() == "scalar_add") {
            auto lhs = get_value<float>("--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs + rhs;
        }

        if (args_.command() == "scalar_sub") {
            auto lhs = get_value<float>("--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs - rhs;
        }

        if (args_.command() == "scalar_mul") {
            auto lhs = get_value<float>("--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs * rhs;
        }

        if (args_.command() == "scalar_div") {
            auto lhs = get_value<float>("--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs / rhs;
        }

        if (args_.command() == "pow") {
            auto self = get_tensor(ctx, "--this");
            auto exponent = get_value<float>("--exponent");

            return self.pow(exponent);
        }

        
        if (args_.command() == "clip") {
            auto self = get_tensor(ctx, "--this");
            auto min = get_value<float>("--min");
            auto max = get_value<float>("--max");

            return self.clip(min, max);
        }

        if (args_.command() == "sum") {
            auto self = get_tensor(ctx, "--this");

            return self.sum();
        }

        if (args_.command() == "mean") {
            auto self = get_tensor(ctx, "--this");

            return self.mean();
        }


        if (args_.command() == "index") {
            auto self = get_tensor(ctx, "--this");
            auto index = get_value<size_t>("--index");

            return self[index];
        }

        if (args_.command() == "slice") {
            auto self = get_tensor(ctx, "--this");
            auto slice = get_slice("--slice");

            return self[slice];
        }


        if (args_.command() == "abs") {
            auto self = get_tensor(ctx, "--this");

            return abs(self);
        }

        if (args_.command() == "sqrt") {
            auto self = get_tensor(ctx, "--this");

            return sqrt(self);
        }

        if (args_.command() == "exp") {
            auto self = get_tensor(ctx, "--this");

            return exp(self);
        }

        if (args_.command() == "log") {
            auto self = get_tensor(ctx, "--this");

            return log(self);
        }

        if (args_.command() == "sin") {
            auto self = get_tensor(ctx, "--this");

            return sin(self);
        }

        if (args_.command() == "cos") {
            auto self = get_tensor(ctx, "--this");

            return cos(self);
        }

        if (args_.command() == "rsqrt") {
            auto self = get_tensor(ctx, "--this");

            return rsqrt(self);
        }

        throw std::runtime_error("Uknown command: " + args_.command());
    }

    void compute(GGMLGraph& graph) {
        GGMLComputation computation(graph);

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
    T get_value(std::string_view option) {
        auto value = args_.get<T>(option).value();

        std::cerr << "argument " << option << ": " << value << std::endl;
        return value;
    }

    template <typename T>
    T get_value(std::string_view option, const T& default_value) {
        auto value = args_.get<T>(option).value_or(default_value);

        std::cerr << "argument " << option << ": " << value << std::endl;
        return value;
    }

    Tensor::Shape get_shape(std::string_view option) {
        auto value = get_value<std::string>(option);
        
        ShapeParser parser(value);

        return parser.parse();
    }

    std::vector<Tensor::Slice> get_slice(std::string_view option) {
        auto value = get_value<std::string>(option);
        
        SliceParser parser(value);

        return parser.parse();
    }

    Tensor get_tensor(GGMLContext& ctx, std::string_view option) {
        auto value = get_value<std::string>(option);

        TensorParser parser(value);

        auto [shape, data] = parser.parse();
        auto tensor = Tensor::empty(*ctx, shape, GGML_TYPE_F32);

        std::cerr << "inferred shape for " << option << ": " << shape.to_string() << std::endl;

        inputs_.push_back({tensor, data});

        return tensor;
    }

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

    GGMLBackend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGMLScheduler scheduler({*cpu});

    TensorCLI tensor_cli(argc, argv);

    auto graph = scheduler.plan(tensor_cli);
    tensor_cli.compute(graph);

    return EXIT_SUCCESS;
}
