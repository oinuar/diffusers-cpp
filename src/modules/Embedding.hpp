#pragma once

#include "modules/Module.hpp"

class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim, std::optional<int64_t> padding_idx = std::nullopt);

    Tensor forward(ggml_context* ctx, Tensor input);
};
