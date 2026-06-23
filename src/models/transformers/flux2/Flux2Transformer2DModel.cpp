#include "models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "GGUFLoaderVisitor.hpp"

Flux2Transformer2DModel Flux2Transformer2DModel::from_pretrained(GGMLBackend& loader_backend, const std::string& path) {
    Flux2Transformer2DModel model(
        1,
        128,
        std::nullopt,
        8,
        24,
        128,
        32,
        12288,
        256,
        3.0,
        {32, 32, 32, 32},
        2000,
        1e-6,
        false
    );

    GGUFLoaderVisitor loader(loader_backend, path);
    model.accept(loader);

    return std::move(model);
}
