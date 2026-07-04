#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <cctype>

#include "Tensor.hpp"

class TensorParser {
public:
    explicit TensorParser(std::string_view s)
        : s_(s), i_(0) {}

    std::pair<Tensor::Shape, std::vector<float>> parse() {
        skip_ws();

        Node result = parse_array();

        skip_ws();

        if (i_ != s_.size())
            throw std::runtime_error("Unexpected trailing characters in tensor literal");

        return {result.shape, std::move(result.values)};
    }

private:
    struct Node {
        std::vector<float> values;
        Tensor::Shape shape;
        bool scalar = false; // true only for parse_number()
    };

    std::string_view s_;
    size_t i_;

    Node parse_array() {
        expect('[');

        std::vector<Node> elements;

        skip_ws();

        if (peek() == ']') {
            consume();

            Node out;
            out.shape = Tensor::Shape{0};
            return out;
        }

        while (true) {
            skip_ws();

            if (peek() == '[')
                elements.push_back(parse_array());
            else
                elements.push_back(parse_number());

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

        while (std::isdigit(static_cast<unsigned char>(peek()))) {
            found = true;
            value = value * 10.0f + (peek() - '0');
            consume();
        }

        if (peek() == '.') {
            consume();

            float frac = 0.1f;

            while (std::isdigit(static_cast<unsigned char>(peek()))) {
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

        Node out;
        out.values.push_back(value);
        out.shape = Tensor::Shape{1};
        out.scalar = true;

        return out;
    }

    Node flatten(const std::vector<Node>& nodes) {
        Node out;

        if (nodes.empty())
            return out;

        // Array of scalars: [1, 2, 3]
        if (nodes[0].scalar) {
            for (const auto& n : nodes) {
                if (!n.scalar)
                    throw std::runtime_error("Mixed scalars and arrays are not supported");

                out.values.push_back(n.values[0]);
            }

            out.shape = Tensor::Shape{static_cast<int64_t>(nodes.size())};
            out.scalar = false;
            return out;
        }

        // Array of arrays: [[...], [...], ...]
        const Tensor::Shape& firstShape = nodes[0].shape;

        for (const auto& n : nodes) {
            if (n.scalar)
                throw std::runtime_error("Mixed scalars and arrays are not supported");

            if (n.shape.rank() != firstShape.rank())
                throw std::runtime_error("Jagged tensor not supported");

            for (int i = 0; i < firstShape.rank(); ++i) {
                if (n.shape[i] != firstShape[i])
                    throw std::runtime_error("Jagged tensor not supported");
            }

            out.values.insert(
                out.values.end(),
                n.values.begin(),
                n.values.end());
        }

        std::vector<int64_t> dims;
        dims.push_back(static_cast<int64_t>(nodes.size()));

        for (int i = 0; i < firstShape.rank(); ++i)
            dims.push_back(firstShape[i]);

        Tensor::Shape shape(dims.size());

        for (size_t i = 0; i < dims.size(); ++i)
            shape[i] = dims[i];

        out.shape = std::move(shape);
        out.scalar = false;

        return out;
    }

    char peek() const {
        if (i_ >= s_.size())
            return '\0';

        return s_[i_];
    }

    char consume() {
        return s_[i_++];
    }

    void expect(char c) {
        if (peek() != c)
            throw std::runtime_error(std::string("Expected '") + c + "'");

        consume();
    }

    void skip_ws() {
        while (i_ < s_.size() &&
               std::isspace(static_cast<unsigned char>(s_[i_])))
            ++i_;
    }
};