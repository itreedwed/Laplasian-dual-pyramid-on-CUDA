#include "filter_cpu.h"
#include "filter_params.h"

#include <algorithm>
#include <vector>

namespace {

inline size_t pixelIndex(int x, int y, int c, int width, int channels) {
    return (static_cast<size_t>(y) * width + x) * channels + c;
}

inline unsigned char saturateToByte(float value) {
    return static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, value)));
}

} // namespace

void applyLaplacianCPU(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels, int apertureK) {
    const LdpFilterParams params = makeLdpFilterParams(apertureK);
    const size_t imageElements = static_cast<size_t>(width) * height * channels;

    std::vector<float> rowForward(imageElements);
    std::vector<float> rowBackward(imageElements);
    std::vector<float> rowFiltered(imageElements);
    std::vector<float> columnForward(imageElements);
    std::vector<float> columnBackward(imageElements);
    std::vector<float> filtered(imageElements);

    // Two directional recursive passes along image rows.
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int c = 0; c < channels; ++c) {
            float previous = static_cast<float>(input[pixelIndex(0, y, c, width, channels)]);
            rowForward[pixelIndex(0, y, c, width, channels)] = previous;
            for (int x = 1; x < width; ++x) {
                const size_t index = pixelIndex(x, y, c, width, channels);
                previous = params.alpha * static_cast<float>(input[index]) + params.beta * previous;
                rowForward[index] = previous;
            }

            previous = static_cast<float>(input[pixelIndex(width - 1, y, c, width, channels)]);
            rowBackward[pixelIndex(width - 1, y, c, width, channels)] = previous;
            for (int x = width - 2; x >= 0; --x) {
                const size_t index = pixelIndex(x, y, c, width, channels);
                previous = params.alpha * static_cast<float>(input[index]) + params.beta * previous;
                rowBackward[index] = previous;
            }
        }
    }

#pragma omp parallel for
    for (int index = 0; index < static_cast<int>(imageElements); ++index) {
        rowFiltered[index] = 0.5f * (rowForward[index] + rowBackward[index]);
    }

    // The same two directional recursive passes along image columns.
#pragma omp parallel for
    for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
            float previous = rowFiltered[pixelIndex(x, 0, c, width, channels)];
            columnForward[pixelIndex(x, 0, c, width, channels)] = previous;
            for (int y = 1; y < height; ++y) {
                const size_t index = pixelIndex(x, y, c, width, channels);
                previous = params.alpha * rowFiltered[index] + params.beta * previous;
                columnForward[index] = previous;
            }

            previous = rowFiltered[pixelIndex(x, height - 1, c, width, channels)];
            columnBackward[pixelIndex(x, height - 1, c, width, channels)] = previous;
            for (int y = height - 2; y >= 0; --y) {
                const size_t index = pixelIndex(x, y, c, width, channels);
                previous = params.alpha * rowFiltered[index] + params.beta * previous;
                columnBackward[index] = previous;
            }
        }
    }

#pragma omp parallel for
    for (int index = 0; index < static_cast<int>(imageElements); ++index) {
        filtered[index] = 0.5f * (columnForward[index] + columnBackward[index]);
        output[index] = saturateToByte(static_cast<float>(input[index]) - filtered[index] + 128.0f);
    }
}
