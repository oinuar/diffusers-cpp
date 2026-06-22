#include "GGMLCompute.hpp"
#include "GGMLComputation.hpp"
#include "GGMLBackend.hpp"
#include "GGMLScheduler.hpp"
#include "models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "ParametersVisitor.hpp"
#include <iostream>

class Flux2Transformer2DModelCompute : public GGMLCompute {
public:
    virtual std::pair<Parameters, Tensor> build(GGMLContext& ctx) {
        Linear model(1, 1);

        // create tensors
        auto x = Tensor::empty<1>(*ctx, GGML_TYPE_F32, {1});

        auto result = model.forward(*ctx, x);
        
        return {{{"x", x}}, result};
    }

    virtual void compute(GGMLGraph& graph) {
        ggml_graph_print(*graph);
        ggml_graph_dump_dot(*graph, NULL, "ggml_graph.dot");
    }
};

int main() {
    Flux2Transformer2DModel model;

    ParametersVisitor visitor;
    model.accept(visitor);

    /*ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    GGMLBackend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGMLScheduler scheduler({*cpu});

    Flux2Transformer2DModelCompute transformer;

    auto graph = scheduler.plan(transformer);
    transformer.compute(graph);*/

    return 0;
}
