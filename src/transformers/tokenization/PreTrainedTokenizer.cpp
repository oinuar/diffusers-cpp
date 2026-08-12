#include "transformers/tokenization/PreTrainedTokenizer.hpp"
#include <fstream>

using namespace tokenizers;

PreTrainedTokenizer::PreTrainedTokenizer(const std::filesystem::path& tokenizer_file) {
    std::ifstream file(tokenizer_file, std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open tokenizer file: " + tokenizer_file.string());

    std::string json(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    tokenizer_ = Tokenizer::FromBlobJSON(json);

    if (!tokenizer_)
        throw std::runtime_error("Failed to create tokenizer from: " + tokenizer_file.string());
}

std::vector<int> PreTrainedTokenizer::encode(
    const std::string& text,
    int max_length,
    std::vector<int>* mask,
    size_t* num_real_tokens) const
{
    auto ids = tokenizer_->Encode(text);

    auto real_tokens = ids.size();

    if (max_length > 0)
        ids.resize(max_length, pad_token_id());

    if (num_real_tokens)
        *num_real_tokens = std::min(
            real_tokens,
            max_length > 0
                ? static_cast<size_t>(max_length)
                : real_tokens);

    if (mask) {
        mask->resize(ids.size());

        auto n = std::min(real_tokens, ids.size());

        for (auto i = 0; i < ids.size(); ++i)
            (*mask)[i] = i < n ? 1 : 0;
    }

    return ids;
}

std::string PreTrainedTokenizer::decode(const std::vector<int>& ids) const {
    return tokenizer_->Decode(ids);
}
