#pragma once

#include "nn/Module.hpp"

class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim, std::optional<int64_t> padding_idx = std::nullopt);

    Tensor forward(Context& context, Tensor input);

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    std::optional<int64_t> padding_idx_;
};
