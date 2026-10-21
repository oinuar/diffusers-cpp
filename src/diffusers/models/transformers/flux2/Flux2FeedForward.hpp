#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"

class Flux2FeedForward : public Module {
private:
    Flux2FeedForward(
        int64_t dim,
        int64_t dim_out,
        float mult,
        int64_t inner_dim,
        bool bias
    ) {
        // The reference fuses the gate and up projections into one
        // Linear (dim -> inner * 2) and chunks its output; the split form
        // is mathematically identical (row-slicing a projection is the
        // same projection) but shardable: on the meta device the fused
        // output cannot be split, because a windowed view of a
        // column-parallel (S(0)) tensor cannot carry the shard (each
        // device holds a different half of the features, so the SwiGLU
        // product would be cross-device). With separate gate/up
        // projections the meta plans them column-parallel (weight S(1)
        // -> output S(0)), the SwiGLU product stays S(0), and linear_out
        // can run row-parallel (weight S(0) + activation S(0) ->
        // PARTIAL -> AllReduce -> R) instead of fully replicated.
        modules["linear_in_gate"] = std::make_shared<Linear>(dim, inner_dim, bias);
        modules["linear_in_up"] = std::make_shared<Linear>(dim, inner_dim, bias);
        modules["act_fn"] = std::make_shared<Flux2SwiGLU>();
        modules["linear_out"] = std::make_shared<Linear>(inner_dim, dim_out, bias);
    }

public:
    Flux2FeedForward(
        int64_t dim,
        std::optional<int64_t> dim_out = std::nullopt,
        float mult = 3.0,
        std::optional<int64_t> inner_dim = std::nullopt,
        bool bias = false
    ) :
        Flux2FeedForward(
            dim,
            dim_out.value_or(dim),
            mult,
            inner_dim.value_or(dim * mult),
            bias
        )
    {
    }

    Tensor forward(Scope scope, Tensor x) {
        auto linear_in_gate = std::static_pointer_cast<Linear>(modules["linear_in_gate"]);
        auto linear_in_up = std::static_pointer_cast<Linear>(modules["linear_in_up"]);
        auto act_fn = std::static_pointer_cast<Flux2SwiGLU>(modules["act_fn"]);
        auto linear_out = std::static_pointer_cast<Linear>(modules["linear_out"]);

        auto gate = linear_in_gate->forward(scope, x);
        auto up = linear_in_up->forward(scope, x);

        x = act_fn->forward(scope, gate, up);
        x = linear_out->forward(scope, x);

        return x;
    }

    std::shared_ptr<Linear> linear_in_gate() const {
        return std::static_pointer_cast<Linear>(modules.at("linear_in_gate"));
    }

    std::shared_ptr<Linear> linear_in_up() const {
        return std::static_pointer_cast<Linear>(modules.at("linear_in_up"));
    }

    std::shared_ptr<Linear> linear_out() const {
        return std::static_pointer_cast<Linear>(modules.at("linear_out"));
    }

    void accept(Visitor& visitor, std::vector<std::string> path = {}) override {
        visitor.visit(*this, std::move(path));
    }
};
