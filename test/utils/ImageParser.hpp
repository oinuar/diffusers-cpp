#pragma once

#include <string>
#include "Image.hpp"
#include "./ArgumentParser.hpp"

template <>
struct ArgumentParser::parser<Image> {
    Image operator ()(const std::string&, const std::string& value) const {
        return std::move(Image::from_file(value));
    }
};
