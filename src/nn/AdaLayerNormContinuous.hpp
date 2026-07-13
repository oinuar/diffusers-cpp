#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/LayerNorm.hpp"
#include "nn/RMSNorm.hpp"
#include <iostream>

class AdaLayerNormContinuous : public Module {
public:
    AdaLayerNormContinuous(
        int64_t embedding_dim,
        int64_t conditioning_embedding_dim,
        bool elementwise_affine = true,
        float eps = 1e-5f,
        bool bias = true,
        const char* norm_type = "layer_norm"
    ) : norm_type_(norm_type) {
        modules["silu"] = std::make_shared<SiLU>();
        modules["linear"] = std::make_shared<Linear>(conditioning_embedding_dim, embedding_dim * 2, bias);

        if (norm_type_ == "layer_norm")
            modules["norm"] = std::make_shared<LayerNorm>(embedding_dim, eps, elementwise_affine, bias);
        else if (norm_type_ == "rms_norm")
            modules["norm"] = std::make_shared<RMSNorm>(embedding_dim, eps, elementwise_affine);
    }

    Tensor forward(ggml_context* ctx, Tensor hidden_states, Tensor conditioning_embedding) {
        auto silu = std::static_pointer_cast<SiLU>(modules["silu"]);
        auto linear = std::static_pointer_cast<Linear>(modules["linear"]);

        auto emb = linear->forward(ctx, silu->forward(ctx, conditioning_embedding));
        auto chunk = emb.chunk(2, 1);
        auto scale = chunk.at(0);
        auto shift = chunk.at(1);

        if (norm_type_ == "layer_norm")
            hidden_states = std::static_pointer_cast<LayerNorm>(modules["norm"])->forward(ctx, hidden_states);
        else if (norm_type_ == "rms_norm")
            hidden_states = std::static_pointer_cast<RMSNorm>(modules["norm"])->forward(ctx, hidden_states);

        // TODO: remove hidden_states[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}] 
        // and make broadcasting (*) work to match PyTorch
        hidden_states = hidden_states[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}] * (1 + scale)[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}] + shift[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}];

        return hidden_states;
    }
private:
    std::string norm_type_;
};
