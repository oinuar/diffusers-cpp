#include "GGMLBackend.hpp"
#include "GGMLScheduler.hpp"
#include "models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include <iostream>

// TODO: not like this, make Flux2KleinPipeline a GGMLCompute instead
/* class Flux2Transformer2DModelCompute : public GGMLCompute {
public:
    Flux2Transformer2DModelCompute(ModelLoaderVisitor& loader) : model_(1, 1) {

        // Load the model & allocate parameters
        model_.accept(loader);
    }

    static void from_pretrained(const std::string& diffusers_path, const std::string& text_encoder_path) {
        GGUFModelLoader loader(diffusers_path);

        Flux2Transformer2DModel transformer;
        transformer.accept(loader);
    }

    virtual Tensor build(GGMLContext& ctx) {
        // create tensors
        auto x = Tensor::empty<1>(*ctx, GGML_TYPE_F32, {1});

        auto result = model_.forward(*ctx, x);

        return result;
    }

    virtual void compute(GGMLGraph& graph) {
        ggml_graph_print(*graph);
        ggml_graph_dump_dot(*graph, NULL, "ggml_graph.dot");
    }
private:
    CollectParametersVisitor visitor_;
    Linear model_;
};*/

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    GGMLBackend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);

    std::string path = "../examples/flux2-cli/utils/convert-model/flux2-klein-9b_q8_0.gguf";

    auto transformer = Flux2Transformer2DModel::from_pretrained(cpu, path);

    /*GGMLScheduler scheduler({*cpu});

    Flux2Transformer2DModelCompute transformer;

    auto graph = scheduler.plan(transformer);
    transformer.compute(graph);*/

    return 0;
}
