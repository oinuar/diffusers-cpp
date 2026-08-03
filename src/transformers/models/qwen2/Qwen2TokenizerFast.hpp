#pragma once

#include <cstdint>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

class Qwen2TokenizerFast {
public:
    struct Message {
        std::string role;
        std::string content;
    };

    explicit Qwen2TokenizerFast(const std::filesystem::path& tokenizer_file);

    std::vector<int> encode(const std::string& text,
                            int max_length = 0,
                            bool add_special_tokens = true,
                            std::vector<int>* attention_mask = nullptr,
                            size_t* num_real_tokens = nullptr) const;

    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = true) const;

    std::string apply_chat_template(const std::vector<Message>& messages,
                                    bool add_generation_prompt = true) const;

    int vocab_size()   const { return static_cast<int>(vocab_.size()); }
    int eos_token_id() const { return eos_token_id_; }
    int unk_token_id() const { return unk_token_id_; }
    // int pad_token_id() const { return pad_token_id_; }

private:
    // BPE with priority queue optimization
    std::vector<std::string> bpe(const std::string& token) const;
    std::vector<int> bpe_encode_token(const std::string& token) const;

    // Pre-tokenization (GPT-4 regex)
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Byte <-> Unicode (GPT-2 mapping)
    static std::string bytes_to_unicode(const std::vector<uint8_t>& bytes);
    static std::vector<uint8_t> unicode_to_bytes(const std::string& str);
    static void init_byte_encoder();

    // Unicode classification
    static bool is_letter(char32_t c);
    static bool is_number(char32_t c);
    static bool is_whitespace(char32_t c);

    // UTF-8 helpers
    static std::u32string utf8_to_utf32(const std::string& s);
    static std::string  utf32_to_utf8(const std::u32string& s);

    // Data
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    std::unordered_map<std::string, int> merges_; // "part1 part2" -> rank
    std::unordered_map<std::string, int> special_tokens_;
    std::unordered_map<int, std::string> reverse_special_tokens_;

    // BPE cache
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    // Byte encoder
    static std::unordered_map<uint8_t, char32_t> byte_encoder_;
    static std::unordered_map<char32_t, uint8_t> byte_decoder_;
    static bool byte_encoder_initialized_;

    // Config
    std::string unk_token_;
    int eos_token_id_ = -1;
    int unk_token_id_ = -1;
    bool add_prefix_space_ = true;
};
