#include "filter_gpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void laplacianKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels,
                                const float* kernel, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = kSize / 2;

    if (x >= width || y >= height) return;

    // Пропускаем границы для простоты
    if (x < half || x >= width - half || y < half || y >= height - half) {
        // Копируем исходный пиксель
        for (int c = 0; c < channels; ++c)
            output[(y * width + x) * channels + c] = input[(y * width + x) * channels + c];
        return;
    }

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                sum += input[( (y + ky) * width + (x + kx) ) * channels + c] *
                       kernel[(ky + half) * kSize + (kx + half)];
            }
        }
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(
            fmaxf(0.0f, fminf(255.0f, sum + 128.0f)));
    }
}

void applyLaplacianGPU(const unsigned char* h_input, unsigned char* h_output,
                       int width, int height, int channels,
                       const float* h_kernel, int kSize) {
    unsigned char *d_input, *d_output;
    float *d_kernel;
    size_t imgSize = width * height * channels * sizeof(unsigned char);
    size_t kSizeBytes = kSize * kSize * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kSizeBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kSizeBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    laplacianKernel<<<grid, block>>>(d_input, d_output, width, height, channels, d_kernel, kSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}