#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <tokenizers_cpp.h>

class PreTrainedTokenizer {
public:
    explicit PreTrainedTokenizer(const std::filesystem::path& tokenizer_file);

    std::vector<int> encode(
        const std::string& text,
        int max_length = 0,
        std::vector<int>* mask = nullptr,
        size_t* num_real_tokens = nullptr) const;

    std::string decode(const std::vector<int>& ids) const;

    virtual int pad_token_id() const = 0;

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};
