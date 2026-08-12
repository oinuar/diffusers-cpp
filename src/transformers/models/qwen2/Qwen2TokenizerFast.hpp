#pragma once

#include "transformers/tokenization/PreTrainedTokenizer.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Qwen2TokenizerFast : public PreTrainedTokenizer {
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

    static Qwen2TokenizerFast from_pretrained(const std::filesystem::path& path);

    std::string apply_chat_template(
        const std::vector<Message>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = false,
        const std::vector<Tool>& tools = {}) const;

    virtual int pad_token_id() const;

private:
    Qwen2TokenizerFast(const std::filesystem::path& tokenizer_file, const std::filesystem::path& config_file);

    int pad_token_id_;
};
