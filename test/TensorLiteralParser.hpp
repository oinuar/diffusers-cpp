#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <iostream>
#include <iomanip>
#include "Tensor.hpp"

class TensorLiteralParser {
public:
    explicit TensorLiteralParser(std::string_view s)
        : s_(s), i_(0) {}

    std::pair<Tensor::Shape, std::vector<float>> parse() {
        skip_ws();
        auto result = parse_array(0);
        skip_ws();

        if (i_ != s_.size())
            throw std::runtime_error("Unexpected trailing characters in tensor literal");

        return {result.shape, std::move(result.values)};
    }

private:
    struct Node {
        std::vector<float> values;
        Tensor::Shape shape;
    };

    std::string_view s_;
    size_t i_;

    Node parse_array(int depth) {
        expect('[');

        std::vector<Node> elements;
        bool first = true;

        skip_ws();

        if (peek() == ']') {
            consume();
            return Node{{}, Tensor::Shape{0}};
        }

        while (true) {
            skip_ws();

            if (peek() == '[') {
                elements.push_back(parse_array(depth + 1));
            } else {
                elements.push_back(parse_number());
            }

            skip_ws();

            if (peek() == ',') {
                consume();
                continue;
            }

            if (peek() == ']') {
                consume();
                break;
            }

            throw std::runtime_error("Expected ',' or ']'");
        }

        return flatten(elements);
    }

    Node parse_number() {
        skip_ws();

        bool neg = false;
        if (peek() == '-') {
            neg = true;
            consume();
        }

        float value = 0.0f;
        bool found = false;
        float frac = 0.1f;

        while (std::isdigit(peek())) {
            found = true;
            value = value * 10 + (peek() - '0');
            consume();
        }

        if (peek() == '.') {
            consume();

            while (std::isdigit(peek())) {
                found = true;
                value += (peek() - '0') * frac;
                frac *= 0.1f;
                consume();
            }
        }

        if (!found)
            throw std::runtime_error("Invalid number in tensor literal");

        if (neg)
            value = -value;

        return Node{{value}, Tensor::Shape{1}};
    }

    Node flatten(const std::vector<Node>& nodes) {
        Node out;

        if (nodes.empty())
            return out;

        bool is_scalar_row = nodes[0].shape.rank() == 1 && nodes[0].shape[0] == 1;

        if (is_scalar_row) {
            for (auto& n : nodes)
                out.values.push_back(n.values[0]);

            out.shape = Tensor::Shape{static_cast<int64_t>(nodes.size())};
            return out;
        }

        int64_t inner_size = nodes[0].values.size();

        for (auto& n : nodes) {
            if ((int64_t)n.values.size() != inner_size)
                throw std::runtime_error("Jagged tensor not supported");

            out.values.insert(out.values.end(), n.values.begin(), n.values.end());
        }

        std::vector<int64_t> shape_vec;
        shape_vec.push_back(nodes.size());

        for (auto d : nodes[0].shape) {
            if (d != 0)
                shape_vec.push_back(d);
        }

        Tensor::Shape shape(shape_vec.size());
        for (size_t i = 0; i < shape_vec.size(); ++i)
            shape[i] = shape_vec[i];

        out.shape = shape;

        return out;
    }

    char peek() const {
        if (i_ >= s_.size()) return '\0';
        return s_[i_];
    }

    char consume() {
        return s_[i_++];
    }

    void expect(char c) {
        if (peek() != c)
            throw std::runtime_error(std::string("Expected ") + c);
        consume();
    }

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(s_[i_]))
            ++i_;
    }
};