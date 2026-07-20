#pragma once

#include "Graph.hpp"
#include "traits.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class Computation {
public:
    Computation(Graph& graph) : graph_(graph) {
        ggml_backend_sched_reset(graph.sched_);
        ggml_backend_sched_alloc_graph(graph.sched_, graph.gf_);
    }

    void execute() {
        ggml_backend_sched_graph_compute(graph_.sched_, graph_.gf_);
    }

    const std::vector<Tensor>& results() const {
        return graph_.tensors_;
    }

    template<typename T>
    void load(Tensor& tensor, const std::vector<T>& values)
    {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "load(): T must be trivially copyable"
        );

        constexpr auto expected = ggml_type_of<T>::value;

        if (tensor.dtype() != expected)
            throw std::invalid_argument("load(): dtype mismatch: expected " + std::string(ggml_type_name(expected)) + ", but got " + std::string(ggml_type_name(tensor.dtype())));

        if (values.size() * sizeof(T) != ggml_nbytes(*tensor))
            throw std::invalid_argument("load(): data size mismatch: expected " + std::to_string(values.size() * sizeof(T)) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_set(
            *tensor,
            values.data(),
            0,
            ggml_nbytes(*tensor)
        );
    }

    template<class T>
    std::vector<T> read(const Tensor& tensor)
    {
        constexpr auto expected = ggml_type_of<T>::value;

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
};
