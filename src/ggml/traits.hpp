#pragma once

#include "ggml.h"

template<typename T>
struct ggml_type_of;

template<>
struct ggml_type_of<float> {
    static constexpr ggml_type value = GGML_TYPE_F32;
};

template<>
struct ggml_type_of<int8_t> {
    static constexpr ggml_type value = GGML_TYPE_I8;
};

template<>
struct ggml_type_of<uint8_t> {
    // GGML has no unsigned byte type.
    // Use I8 storage.
    static constexpr ggml_type value = GGML_TYPE_I8;
};

template<>
struct ggml_type_of<int16_t> {
    static constexpr ggml_type value = GGML_TYPE_I16;
};

template<>
struct ggml_type_of<int32_t> {
    static constexpr ggml_type value = GGML_TYPE_I32;
};

template<>
struct ggml_type_of<int64_t> {
    static constexpr ggml_type value = GGML_TYPE_I64;
};
