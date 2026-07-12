#pragma once

#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ggml/Tensor.hpp"
#include "./ArgumentParser.hpp"

template <>
struct ArgumentParser::parser<std::vector<Tensor::Slice>> {
    std::vector<Tensor::Slice> operator ()(const std::string&, const std::string& value) const {
        SliceParser parser(value);

        return parser.parse();
    }

private:
    class SliceParser {
    public:
        explicit SliceParser(std::string_view text)
            : text_(text), pos_(0) {}

        std::vector<Tensor::Slice> parse() {
            skipWhitespace();

            if (peek() == '[')
                consume();

            std::vector<Tensor::Slice> result;

            while (true) {
                skipWhitespace();

                if (eof())
                    break;

                if (peek() == ']') {
                    consume();
                    break;
                }

                result.push_back(parseSlice());

                skipWhitespace();

                if (peek() == ',') {
                    consume();
                    continue;
                }

                if (peek() == ']')
                    continue;

                if (!eof())
                    throw std::runtime_error("Expected ',' or ']'");
            }

            return result;
        }

    private:
        std::string_view text_;
        size_t pos_;

        bool eof() const {
            return pos_ >= text_.size();
        }

        char peek() const {
            return eof() ? '\0' : text_[pos_];
        }

        char consume() {
            return text_[pos_++];
        }

        void skipWhitespace() {
            while (!eof() && std::isspace((unsigned char)peek()))
                consume();
        }

        bool match(std::string_view s) {
            if (text_.substr(pos_, s.size()) == s) {
                pos_ += s.size();
                return true;
            }
            return false;
        }

        std::optional<int64_t> parseOptionalInt() {
            skipWhitespace();

            if (peek() == ':' || peek() == ',' || peek() == ']')
                return std::nullopt;

            bool neg = false;

            if (peek() == '-') {
                neg = true;
                consume();
            }

            if (!std::isdigit(peek()))
                throw std::runtime_error("Expected integer");

            int64_t value = 0;

            while (std::isdigit(peek()))
                value = value * 10 + (consume() - '0');

            return neg ? -value : value;
        }

        Tensor::Slice parseSlice() {
            skipWhitespace();

            if (match("None"))
                return Tensor::Slice::none();

            if (match("..."))
                return Tensor::Slice::ellipsis();

            // Starts with ':' -> all() or range()
            if (peek() == ':') {
                consume();

                auto stop = parseOptionalInt();

                if (peek() != ':') {
                    if (!stop)
                        return Tensor::Slice::all();

                    return Tensor::Slice::range(std::nullopt, stop);
                }

                consume();

                auto step = parseOptionalInt();

                return Tensor::Slice::range(std::nullopt, stop, step.value());
            }

            auto first = parseOptionalInt();

            skipWhitespace();

            if (peek() != ':')
                return Tensor::Slice::index(*first);

            consume();

            auto stop = parseOptionalInt();

            if (peek() != ':')
                return Tensor::Slice::range(first, stop);

            consume();

            auto step = parseOptionalInt();

            return Tensor::Slice::range(first, stop, step.value());
        }
    };
};
