#include "Image.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <cmath>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize2.h>

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

Image Image::from_file(const std::filesystem::path& path) {
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

Image Image::resize(size_t width, size_t height) const {
    if (width == 0 || height == 0)
        throw std::invalid_argument(
            "Image::resize(): dimensions must be greater than zero");

    if (width == width_ && height == height_)
        return *this;

    std::vector<uint8_t> resized(
        width * height * channels_);

    const stbir_pixel_layout layout =
        channels_ == 1 ? STBIR_1CHANNEL :
        channels_ == 2 ? STBIR_RA :
        channels_ == 3 ? STBIR_RGB :
        channels_ == 4 ? STBIR_RGBA :
        throw std::invalid_argument(
            "Image::resize(): unsupported channel count: " +
            std::to_string(channels_));

    const auto result = stbir_resize_uint8_srgb(
        pixels_.data(),
        static_cast<int>(width_),
        static_cast<int>(height_),
        static_cast<int>(width_ * channels_),
        resized.data(),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(width * channels_),
        layout
    );

    if (!result)
        throw std::runtime_error(
            "Image::resize(): stb_image_resize failed");

    return Image(
        width,
        height,
        channels_,
        std::move(resized)
    );
}

Image Image::resize_and_crop(size_t width, size_t height) const {
    if (width == 0 || height == 0)
        throw std::invalid_argument(
            "Image::resize_and_crop(): dimensions must be greater than zero");

    if (width_ == width && height_ == height)
        return *this;

    // Scale so that the requested rectangle is completely covered.
    const double scale = std::max(
        static_cast<double>(width) / width_,
        static_cast<double>(height) / height_);

    const size_t resized_width =
        std::max<size_t>(
            1,
            static_cast<size_t>(
                std::round(width_ * scale)));

    const size_t resized_height =
        std::max<size_t>(
            1,
            static_cast<size_t>(
                std::round(height_ * scale)));

    Image resized =
        resize(resized_width, resized_height);

    // Center crop.
    const size_t left =
        (resized_width - width) / 2;

    const size_t top =
        (resized_height - height) / 2;

    std::vector<uint8_t> cropped(
        width * height * channels_);

    for (size_t y = 0; y < height; ++y) {
        const uint8_t* src =
            resized.pixels_.data() +
            ((top + y) * resized_width + left) * channels_;

        uint8_t* dst =
            cropped.data() +
            y * width * channels_;

        std::copy(
            src,
            src + width * channels_,
            dst);
    }

    return Image(
        width,
        height,
        channels_,
        std::move(cropped)
    );
}
