#include "diffusers/models/normalization/AdaLayerNormContinuous.hpp"
#include "ggml/Context.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"

template <class NormFn>
AdaLayerNormContinuous<NormFn>::AdaLayerNormContinuous(
    int64_t embedding_dim,
    int64_t conditioning_embedding_dim,
    bool elementwise_affine,
    float eps,
    bool bias
) {
    modules["silu"] = std::make_shared<SiLU>();
    modules["linear"] = std::make_shared<Linear>(conditioning_embedding_dim, embedding_dim * 2, bias);

    if constexpr (std::is_same_v<NormFn, LayerNorm>)
        modules["norm"] = std::make_shared<NormFn>(embedding_dim, eps, elementwise_affine, bias);
    else if constexpr (std::is_same_v<NormFn, RMSNorm>)
        modules["norm"] = std::make_shared<NormFn>(embedding_dim, eps, elementwise_affine);
}

template <class NormFn>
Tensor AdaLayerNormContinuous<NormFn>::forward(Scope scope, Tensor hidden_states, Tensor conditioning_embedding) {
    auto silu = std::static_pointer_cast<SiLU>(modules["silu"]);
    auto linear = std::static_pointer_cast<Linear>(modules["linear"]);

    auto emb = linear->forward(scope, silu->forward(scope, conditioning_embedding));
    auto chunk = emb.chunk(2, 1);
    auto scale = chunk.at(0);
    auto shift = chunk.at(1);

    auto it = modules.find("norm");

    if (it != std::end(modules))
        hidden_states = std::static_pointer_cast<NormFn>(it->second)->forward(scope, hidden_states);

    hidden_states = hidden_states * (1.0f + scale)[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}] + shift[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}];

    return hidden_states;
}
