#pragma once

#include "ggml/Compute.hpp"
#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "./ArgumentParser.hpp"
#include <iostream>

class TestCLI : public Compute {
public:
    int main() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

        ggml_backend_load_all();

        Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
        Scheduler scheduler({*cpu});

        auto graph = scheduler.plan(*this);
        compute(graph);

        return EXIT_SUCCESS;
    }

    void compute(Graph& graph) {
        Computation computation(graph);

        for (auto& tensor : computation.results()) {
            switch (tensor.dtype()) {
            case Tensor::TypeOf<float>::value:
            {
                auto data = computation.read<float>(tensor);
                print_tensor(data, tensor.shape());
                break;
            }

            case Tensor::TypeOf<int8_t>::value:
            {
                auto data = computation.read<int8_t>(tensor);
                print_tensor(data, tensor.shape());
                break;
            }

            case Tensor::TypeOf<int16_t>::value:
            {
                auto data = computation.read<int16_t>(tensor);
                print_tensor(data, tensor.shape());
                break;
            }

            case Tensor::TypeOf<int32_t>::value:
            {
                auto data = computation.read<int32_t>(tensor);
                print_tensor(data, tensor.shape());
                break;
            }

            default:
                throw std::runtime_error("Unsupported tensor type: " + std::string(ggml_type_name(tensor.dtype())));
            }
        }
    }

protected:
    ArgumentParser args_;
    std::vector<std::pair<Tensor, std::vector<float>>> inputs_;

    TestCLI(int argc, char** argv) : args_(argc, argv) {}

private:
    template <typename T>
    void print_tensor(const std::vector<T>& data, const Tensor::Shape& shape) {
        size_t expected = 1;
        for (auto i = 0; i < shape.rank(); ++i)
            expected *= shape[i];

        if (expected != data.size())
            throw std::runtime_error("tensor data size does not match shape");

        std::cerr << "output shape: " << shape.to_string() << std::endl;

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
