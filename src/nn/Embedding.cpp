#include "nn/Embedding.hpp"
#include "nn/Parameter.hpp"

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, std::optional<int> padding_idx)
{
    modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({embedding_dim, num_embeddings}));
}

Tensor Embedding::forward(ggml_context* ctx, Tensor input) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    // There are issues with ggml batch inference, so we are expanding it here first.
    // TODO: fix ggml batch inference
    int64_t n = inputs.shape()[1];
    auto input_ids = ggml_reshape_1d(ctx, *inputs, inputs.shape()[0] * inputs.shape()[1]);

    input_ids      = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
    auto embedding = ggml_get_rows(ctx, *weight, input_ids);
    embedding      = ggml_reshape_3d(ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

    // [N, n_token, embedding_dim]
    return Tensor(ctx, embedding);
}
