#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

class Image {
public:
    Image(
        size_t width,
        size_t height,
        size_t channels,
        std::vector<uint8_t> pixels
    );

    static Image load(const std::filesystem::path& path);

    void save(const std::filesystem::path& path, int quality = 95) const;

    size_t width() const {
        return width_;
    }

    size_t height() const {
        return height_;
    }

    size_t channels() const {
        return channels_;
    }

    const std::vector<uint8_t>& pixels() const {
        return pixels_;
    }

    std::vector<uint8_t>& pixels() {
        return pixels_;
    }

private:
    size_t width_;
    size_t height_;
    size_t channels_;
    std::vector<uint8_t> pixels_;
};
