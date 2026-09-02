#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"

class Flux2Modulation : public Module {
public:
    Flux2Modulation(int64_t dim, int64_t mod_param_sets = 2, bool bias = false) {
        modules["linear"] = std::make_shared<Linear>(dim, dim * 3 * mod_param_sets, bias);
        modules["act_fn"] = std::make_shared<SiLU>();
    }

    Tensor forward(Scope scope, Tensor temb) {
        auto act_fn = std::static_pointer_cast<SiLU>(modules["act_fn"]);
        auto linear = std::static_pointer_cast<Linear>(modules["linear"]);

        auto mod = act_fn->forward(scope, temb);
        mod = linear->forward(scope, mod);

        return mod;
    }

    static std::vector<std::array<Tensor, 3>> split(Tensor mod, int64_t mod_param_sets) {
        if (mod.ndim() == 2)
            mod = mod.unsqueeze(1);

        auto mod_params = mod.chunk(3 * mod_param_sets, -1);
        std::vector<std::array<Tensor, 3>> result;

        // Return vector of 3-array of modulation params shift/scale/gate
        for (auto i = 0; i < mod_param_sets; ++i) {
            result.push_back({
                mod_params.at(3 * i),
                mod_params.at(3 * i + 1),
                mod_params.at(3 * i + 2)
            });
        }

        return std::move(result);
    }
};
