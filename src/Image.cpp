#include "Image.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>

Image::Image(
    size_t width,
    size_t height,
    size_t channels,
    std::vector<uint8_t> pixels
) :
    width_(width),
    height_(height),
    channels_(channels),
    pixels_(std::move(pixels))
{
    if (width_ == 0 || height_ == 0)
        throw std::invalid_argument(
            "Image: invalid dimensions"
        );

    if (channels_ == 0 || channels_ > 4)
        throw std::invalid_argument(
            "Image: unsupported channel count"
        );

    const size_t expected =
        width_ *
        height_ *
        channels_;

    if (pixels_.size() != expected)
        throw std::invalid_argument(
            "Image: pixel buffer size mismatch"
        );
}

Image Image::load(const std::filesystem::path& path) {
    int width;
    int height;
    int channels;

    auto pixels = stbi_load(
        path.string().c_str(),
        &width,
        &height,
        &channels,
        0
    );


    if (!pixels)
        throw std::runtime_error(
            "Image::load(): " +
            std::string(stbi_failure_reason())
        );


    auto size =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        static_cast<size_t>(channels);


    std::vector<uint8_t> data(
        pixels,
        pixels + size
    );


    stbi_image_free(pixels);

    return Image(
        width,
        height,
        channels,
        std::move(data)
    );
}


void Image::save(const std::filesystem::path& path, int quality) const {
    auto ext =
        path.extension().string();

    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](unsigned char c) {
            return static_cast<char>(
                std::tolower(c)
            );
        });

    const int stride =
        width_ * channels_;

    int ok = 0;

    if (ext == ".png")
    {
        ok = stbi_write_png(
            path.string().c_str(),
            width_,
            height_,
            channels_,
            pixels_.data(),
            stride
        );
    }
    else if (
        ext == ".jpg" ||
        ext == ".jpeg")
    {
        if (channels_ != 3)
            throw std::invalid_argument(
                "Image::save(): JPEG requires RGB"
            );

        ok = stbi_write_jpg(
            path.string().c_str(),
            width_,
            height_,
            channels_,
            pixels_.data(),
            quality
        );
    }
    else if (ext == ".bmp")
    {
        ok = stbi_write_bmp(
            path.string().c_str(),
            width_,
            height_,
            channels_,
            pixels_.data()
        );
    }
    else if (ext == ".tga")
    {
        ok = stbi_write_tga(
            path.string().c_str(),
            width_,
            height_,
            channels_,
            pixels_.data()
        );
    }
    else
    {
        throw std::invalid_argument(
            "Image::save(): unsupported extension " +
            ext
        );
    }

    if (!ok)
        throw std::runtime_error(
            "Image::save(): failed to write image"
        );
}
