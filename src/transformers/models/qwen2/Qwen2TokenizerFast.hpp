#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "tokenizers_cpp.h"

class Qwen2TokenizerFast {
public:
    struct ToolCall {
        std::string name;
        std::string arguments;
    };

    struct Message {
        std::string role;
        std::string content;

        std::string reasoning_content;
        std::vector<ToolCall> tool_calls;
    };

    struct Tool {
        // JSON representation of the tool, exactly as it should appear
        // inside <tools>.
        std::string json;
    };

    explicit Qwen2TokenizerFast(
        const std::filesystem::path& tokenizer_file);

    std::vector<int> encode(
        const std::string& text,
        int max_length = 0,
        std::vector<int>* mask = nullptr,
        size_t* num_real_tokens = nullptr) const;

    std::string decode(const std::vector<int>& ids) const;

    std::string apply_chat_template(
        const std::vector<Message>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = false,
        const std::vector<Tool>& tools = {}) const;

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    int eos_token_id_;
    int unk_token_id_;
    int pad_token_id_;
};
