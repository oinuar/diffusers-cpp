#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"

class Flux2FusedQKVProjection : public Module {
public:
    Flux2FusedQKVProjection(
        int64_t query_dim,
        int64_t inner_dim,
        int64_t mlp_hidden_dim,
        float mlp_mult_factor,
        bool bias)
        :
        query_dim_(query_dim),
        inner_dim_(inner_dim),
        mlp_hidden_dim_(mlp_hidden_dim),
        mlp_mult_factor_(mlp_mult_factor)
    {
        // QKV projections + MLP input projection. The diffusers reference
        // fuses these into one Linear (to_qkv_mlp_proj); we keep them as
        // separate projections: on the meta device every q/k/v must be the
        // direct output of its own row-parallel projection (weight S(1) ->
        // output S(0)), because the heads shard that the meta flash_attn_ext
        // hard-asserts cannot be carried through a windowed view of a fused
        // output (a view only inherits a shard when it covers the sharded
        // axis in full from its start). The math is identical: row-slicing a
        // projection is the same projection.
        modules["q"] = std::make_shared<Linear>(query_dim_, inner_dim_, bias);
        modules["k"] = std::make_shared<Linear>(query_dim_, inner_dim_, bias);
        modules["v"] = std::make_shared<Linear>(query_dim_, inner_dim_, bias);
        modules["mlp_in"] = std::make_shared<Linear>(query_dim_, mlp_hidden_dim_ * mlp_mult_factor_, bias);
    }

    std::tuple<Tensor, Tensor, Tensor, Tensor> forward(Scope scope, Tensor hidden_states) {
        auto query = q()->forward(scope, hidden_states);
        auto key = k()->forward(scope, hidden_states);
        auto value = v()->forward(scope, hidden_states);
        auto mlp_hidden_states = mlp_in()->forward(scope, hidden_states);

        return {query, key, value, mlp_hidden_states};
    }

    int64_t inner_dim() const {
        return inner_dim_;
    }

    int64_t mlp_out_dim() const {
        return mlp_hidden_dim_ * mlp_mult_factor_;
    }

    std::shared_ptr<Linear> q() const {
        return std::static_pointer_cast<Linear>(modules.at("q"));
    }

    std::shared_ptr<Linear> k() const {
        return std::static_pointer_cast<Linear>(modules.at("k"));
    }

    std::shared_ptr<Linear> v() const {
        return std::static_pointer_cast<Linear>(modules.at("v"));
    }

    std::shared_ptr<Linear> mlp_in() const {
        return std::static_pointer_cast<Linear>(modules.at("mlp_in"));
    }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    int64_t query_dim_;
    int64_t inner_dim_;
    int64_t mlp_hidden_dim_;
    float mlp_mult_factor_;
};
