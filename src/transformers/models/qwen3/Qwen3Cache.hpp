#pragma once

#include <utility>
#include <vector>
#include "ggml/Tensor.hpp"

/** @brief Simple KV cache for Qwen3 attention layers.
 *
 * Stores key/value states per layer index and supports appending new KV pairs along the sequence dimension.
 */
class Qwen3Cache {
public:
    /** @brief Appends new key/value states for the given layer index. */
    void update(int64_t layer_idx, Tensor key, Tensor value) {
        while (past_kv_.size() <= static_cast<size_t>(layer_idx)) {
            past_kv_.push_back({});
        }

        if (past_kv_[static_cast<size_t>(layer_idx)].first.ndim() == 0) {
            // First update: store directly.
            past_kv_[static_cast<size_t>(layer_idx)] = {key, value};
        } else {
            // Concatenate along sequence dimension (dim=2).
            auto new_key   = Tensor::cat({past_kv_[static_cast<size_t>(layer_idx)].first, key}, 2);
            auto new_value = Tensor::cat({past_kv_[static_cast<size_t>(layer_idx)].second, value}, 2);
            past_kv_[static_cast<size_t>(layer_idx)] = {new_key, new_value};
        }
    }

    /** @brief Returns the sequence length from the last layer's KV cache. */
    int64_t get_seq_length() const {
        if (past_kv_.empty()) return 0;

        auto it = std::find_if(past_kv_.rbegin(), past_kv_.rend(),
            [](const std::pair<Tensor, Tensor>& kv) { return kv.first.ndim() > 0; });

        if (it == past_kv_.rend()) return 0;

        return it->first.shape()[2]; // sequence dimension
    }

    /** @brief Returns the key/value pair for a given layer index. */
    std::pair<Tensor, Tensor> get(int64_t layer_idx) const {
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= past_kv_.size()) {
            return {{}, {}};
        }

        return past_kv_[static_cast<size_t>(layer_idx)];
    }

private:
    std::vector<std::pair<Tensor, Tensor>> past_kv_; // vector of (key, value) pairs per layer. Empty tensors indicate no cached data.
};
