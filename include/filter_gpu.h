#pragma once

// Рассчитывает параметры рекурсивного фильтра ЛДП и загружает их в CUDA
// constant memory. Функция должна вызываться до запуска CUDA-ядер ЛДП.
void calculateFilterParams(int apertureK);

struct GpuTimings {
    float kernelMs = 0.0f; // CUDA kernels only
    float totalMs = 0.0f;  // allocation, host<->device transfers and kernels
};

void applyLaplacianGPU(const unsigned char* h_input, unsigned char* h_output,
                       int width, int height, int channels,
                       int apertureK, GpuTimings* timings = nullptr);
