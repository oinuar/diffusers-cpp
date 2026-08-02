#include "nn/Embedding.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Runtime.hpp"

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, std::optional<int64_t> padding_idx)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      padding_idx_(padding_idx)
{    
    if (padding_idx_) {
        int64_t idx = *padding_idx_;
        if (idx > 0) {
            if (idx >= num_embeddings_) {
                throw std::invalid_argument("Padding_idx must be within num_embeddings");
            }
        } else if (idx < 0) {
            if (idx < -num_embeddings_) {
                throw std::invalid_argument("Padding_idx must be within num_embeddings");
            }
            padding_idx_ = num_embeddings_ + idx;
        }
    }

    modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({num_embeddings_, embedding_dim_}));
}

Tensor Embedding::forward(Runtime& runtime, Tensor input) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    if (padding_idx_) {
        std::vector<Tensor> slices;

        // Rows on the left
        if (*padding_idx_ > 0)
            slices.push_back(weight[{Tensor::Slice::range(0, *padding_idx_)}]);

        // Zeroed middle row
        slices.push_back(Tensor::zeros(*runtime.context(), {1, embedding_dim_}));

        // Rows on the right
        if (*padding_idx_ + 1 < num_embeddings_)
            slices.push_back(weight[{Tensor::Slice::range(*padding_idx_ + 1, num_embeddings_)}]);

        weight = Tensor::cat(slices, 0);
    }

    auto input_shape = input.shape();

    // ggml_get_rows expects int32 indices
    if (input.dtype() != GGML_TYPE_I32)
        input = input.to(GGML_TYPE_I32);

    // Flatten indices for lookup
    if (!ggml_is_vector(*input))
        input = input.flatten();

    auto lookup_result = ggml_get_rows(
        *runtime.context(),
        *weight,
        *input
    );

    // Restore original dimensions + embedding dimension
    auto output_shape = Tensor::Shape(input_shape.rank() + 1);

    for (int i = 0; i < input_shape.rank(); ++i)
        output_shape[i] = input_shape[i];

    output_shape[input_shape.rank()] = embedding_dim_;

    return Tensor(*runtime.context(), lookup_result).reshape(output_shape);
}
