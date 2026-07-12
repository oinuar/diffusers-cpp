#pragma once

#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "ggml/Tensor.hpp"
#include "./ArgumentParser.hpp"

template <>
struct ArgumentParser::parser<Tensor::Shape> {
    Tensor::Shape operator ()(const std::string&, const std::string& value) const {
        ShapeParser parser(value);

        return parser.parse();
    }
private:
    class ShapeParser {
    public:
        explicit ShapeParser(const std::string& input)
            : input_(input), pos_(0) {}

        Tensor::Shape parse() {
            skipWhitespace();
            expect('(');

            std::vector<int> result;

            skipWhitespace();
            if (peek() != ')') {
                result.push_back(parseInteger());

                skipWhitespace();
                while (match(',')) {
                    result.push_back(parseInteger());
                    skipWhitespace();
                }
            }

            expect(')');
            skipWhitespace();

            if (pos_ != input_.size()) {
                throw std::runtime_error("Unexpected trailing input");
            }

            Tensor::Shape shape(result.size());

            for (auto i = 0; i < shape.rank(); ++i)
                shape[i] = result[i];

            return shape;
        }

    private:
        const std::string& input_;
        size_t pos_;

        char peek() const {
            if (pos_ >= input_.size())
                return '\0';
            return input_[pos_];
        }

        bool match(char c) {
            skipWhitespace();
            if (peek() == c) {
                ++pos_;
                return true;
            }
            return false;
        }

        void expect(char c) {
            skipWhitespace();
            if (peek() != c) {
                throw std::runtime_error(
                    std::string("Expected '") + c + "'");
            }
            ++pos_;
        }

        void skipWhitespace() {
            while (pos_ < input_.size() &&
                std::isspace(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }

        int parseInteger() {
            skipWhitespace();

            bool negative = false;
            if (peek() == '+' || peek() == '-') {
                negative = (peek() == '-');
                ++pos_;
            }

            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                throw std::runtime_error("Expected integer");
            }

            int value = 0;
            while (std::isdigit(static_cast<unsigned char>(peek()))) {
                value = value * 10 + (peek() - '0');
                ++pos_;
            }

            return negative ? -value : value;
        }
    };

};
