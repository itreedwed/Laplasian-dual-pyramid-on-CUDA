#include "filter_cpu.h"
#include <algorithm>
#include <cstring>

void applyLaplacianCPU(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels,
                       const float* kernel, int kSize) {
    const int half = kSize / 2;
    // Обрабатываем каждый канал отдельно
    for (int c = 0; c < channels; ++c) {
        const unsigned char* in = input + c;
        unsigned char* out = output + c;
        for (int y = half; y < height - half; ++y) {
            for (int x = half; x < width - half; ++x) {
                float sum = 0.0f;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        sum += in[(y + ky) * width * channels + (x + kx) * channels] * 
                               kernel[(ky + half) * kSize + (kx + half)];
                    }
                }
                // Клиппинг в [0, 255]
                out[y * width * channels + x * channels] = static_cast<unsigned char>(
                    std::max(0.0f, std::min(255.0f, sum + 128.0f))); // +128 для сдвига яркости (опционально)
            }
        }
    }
}