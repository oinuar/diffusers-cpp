#pragma once

#include "Graph.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

class Computation {
public:
    Computation(Graph& graph) : graph_(graph) {
        ggml_backend_sched_reset(graph.sched_);
        ggml_backend_sched_alloc_graph(graph.sched_, graph.gf_);

        for (auto& [tensor, initializer] : graph_.initializers_)
            load(tensor, initializer(tensor, graph.rng_));

        ggml_backend_sched_graph_compute(graph_.sched_, graph_.gf_);
    }

    const std::vector<Tensor>& results() const {
        return graph_.tensors_;
    }

    template<class T>
    std::vector<T> read(const Tensor& tensor)
    {
        constexpr auto expected = Tensor::TypeOf<T>::value;

        if (tensor.dtype() != expected)
            throw std::invalid_argument("read(): dtype mismatch: expected " + std::string(ggml_type_name(expected)) + ", but got " + std::string(ggml_type_name(tensor.dtype())));

        std::vector<T> data(
            ggml_nelements(*tensor)
        );

        if (data.size() * sizeof(T) != ggml_nbytes(*tensor))
            throw std::invalid_argument("read(): data size mismatch: expected " + std::to_string(data.size() * sizeof(T)) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_get(
            *tensor,
            data.data(),
            0,
            ggml_nbytes(*tensor)
        );

        return std::move(data);
    }

    Computation(Computation&) = delete;
    Computation(Computation&&) = delete;
    Computation& operator =(const Computation&) = delete;
    Computation& operator =(Computation&&) = delete;

private:
    Graph& graph_;

    void load(Tensor& tensor, const std::vector<std::byte>& bytes) {
        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::invalid_argument("load(): data size mismatch: expected " + std::to_string(bytes.size()) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_set(
            *tensor,
            bytes.data(),
            0,
            ggml_nbytes(*tensor)
        );
    }
};
