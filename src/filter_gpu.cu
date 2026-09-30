#include "filter_gpu.h"
#include "filter_params.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// A single device-wide copy of the recursive coefficients.
__constant__ float d_alpha;
__constant__ float d_beta;

void calculateFilterParams(int apertureK) {
    const LdpFilterParams params = makeLdpFilterParams(apertureK);

    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha, &params.alpha, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_beta, &params.beta, sizeof(float)));
}

__global__ void kernelRowForward(const unsigned char* input, float* output,
                                 int width, int height, int channels) {
    const int workItem = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowCount = height * channels;
    if (workItem >= rowCount) return;

    const int y = workItem / channels;
    const int c = workItem % channels;
    size_t index = (static_cast<size_t>(y) * width) * channels + c;
    float previous = static_cast<float>(input[index]);
    output[index] = previous;

    for (int x = 1; x < width; ++x) {
        index += channels;
        previous = d_alpha * static_cast<float>(input[index]) + d_beta * previous;
        output[index] = previous;
    }
}

__global__ void kernelRowBackward(const unsigned char* input, float* output,
                                  int width, int height, int channels) {
    const int workItem = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowCount = height * channels;
    if (workItem >= rowCount) return;

    const int y = workItem / channels;
    const int c = workItem % channels;
    size_t index = (static_cast<size_t>(y) * width + width - 1) * channels + c;
    float previous = static_cast<float>(input[index]);
    output[index] = previous;

    for (int x = width - 2; x >= 0; --x) {
        index -= channels;
        previous = d_alpha * static_cast<float>(input[index]) + d_beta * previous;
        output[index] = previous;
    }
}

__global__ void kernelAverage(const float* first, const float* second,
                              float* output, size_t elementCount) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elementCount) output[index] = 0.5f * (first[index] + second[index]);
}

__global__ void kernelColumnForward(const float* input, float* output,
                                    int width, int height, int channels) {
    const int workItem = blockIdx.x * blockDim.x + threadIdx.x;
    const int columnCount = width * channels;
    if (workItem >= columnCount) return;

    const int x = workItem / channels;
    const int c = workItem % channels;
    size_t index = static_cast<size_t>(x) * channels + c;
    float previous = input[index];
    output[index] = previous;

    const size_t rowStride = static_cast<size_t>(width) * channels;
    for (int y = 1; y < height; ++y) {
        index += rowStride;
        previous = d_alpha * input[index] + d_beta * previous;
        output[index] = previous;
    }
}

__global__ void kernelColumnBackward(const float* input, float* output,
                                     int width, int height, int channels) {
    const int workItem = blockIdx.x * blockDim.x + threadIdx.x;
    const int columnCount = width * channels;
    if (workItem >= columnCount) return;

    const int x = workItem / channels;
    const int c = workItem % channels;
    const size_t rowStride = static_cast<size_t>(width) * channels;
    size_t index = static_cast<size_t>(height - 1) * rowStride + x * channels + c;
    float previous = input[index];
    output[index] = previous;

    for (int y = height - 2; y >= 0; --y) {
        index -= rowStride;
        previous = d_alpha * input[index] + d_beta * previous;
        output[index] = previous;
    }
}

__global__ void kernelLaplacian(const unsigned char* input, const float* smoothed,
                                unsigned char* output, size_t elementCount) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;

    const float response = static_cast<float>(input[index]) - smoothed[index] + 128.0f;
    output[index] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, response)));
}

void applyLaplacianGPU(const unsigned char* h_input, unsigned char* h_output,
                       int width, int height, int channels, int apertureK,
                       GpuTimings* timings) {
    cudaEvent_t totalStart = nullptr;
    cudaEvent_t totalStop = nullptr;
    cudaEvent_t kernelStart = nullptr;
    cudaEvent_t kernelStop = nullptr;
    CUDA_CHECK(cudaEventCreate(&totalStart));
    CUDA_CHECK(cudaEventCreate(&totalStop));
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));
    CUDA_CHECK(cudaEventRecord(totalStart));

    // Coefficients are calculated on the CPU and copied before every kernel launch sequence.
    calculateFilterParams(apertureK);

    const size_t elementCount = static_cast<size_t>(width) * height * channels;
    const size_t byteCount = elementCount * sizeof(unsigned char);
    const size_t floatByteCount = elementCount * sizeof(float);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_rowForward = nullptr;
    float* d_rowBackward = nullptr;
    float* d_rowFiltered = nullptr;
    float* d_columnForward = nullptr;
    float* d_columnBackward = nullptr;
    float* d_smoothed = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, byteCount));
    CUDA_CHECK(cudaMalloc(&d_output, byteCount));
    CUDA_CHECK(cudaMalloc(&d_rowForward, floatByteCount));
    CUDA_CHECK(cudaMalloc(&d_rowBackward, floatByteCount));
    CUDA_CHECK(cudaMalloc(&d_rowFiltered, floatByteCount));
    CUDA_CHECK(cudaMalloc(&d_columnForward, floatByteCount));
    CUDA_CHECK(cudaMalloc(&d_columnBackward, floatByteCount));
    CUDA_CHECK(cudaMalloc(&d_smoothed, floatByteCount));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, byteCount, cudaMemcpyHostToDevice));

    constexpr int threadsPerBlock = 256;
    const int rowBlocks = (height * channels + threadsPerBlock - 1) / threadsPerBlock;
    const int columnBlocks = (width * channels + threadsPerBlock - 1) / threadsPerBlock;
    const int elementBlocks = static_cast<int>((elementCount + threadsPerBlock - 1) / threadsPerBlock);

    CUDA_CHECK(cudaEventRecord(kernelStart));
    kernelRowForward<<<rowBlocks, threadsPerBlock>>>(d_input, d_rowForward, width, height, channels);
    kernelRowBackward<<<rowBlocks, threadsPerBlock>>>(d_input, d_rowBackward, width, height, channels);
    kernelAverage<<<elementBlocks, threadsPerBlock>>>(d_rowForward, d_rowBackward, d_rowFiltered, elementCount);
    kernelColumnForward<<<columnBlocks, threadsPerBlock>>>(d_rowFiltered, d_columnForward, width, height, channels);
    kernelColumnBackward<<<columnBlocks, threadsPerBlock>>>(d_rowFiltered, d_columnBackward, width, height, channels);
    kernelAverage<<<elementBlocks, threadsPerBlock>>>(d_columnForward, d_columnBackward, d_smoothed, elementCount);
    kernelLaplacian<<<elementBlocks, threadsPerBlock>>>(d_input, d_smoothed, d_output, elementCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, byteCount, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_rowForward));
    CUDA_CHECK(cudaFree(d_rowBackward));
    CUDA_CHECK(cudaFree(d_rowFiltered));
    CUDA_CHECK(cudaFree(d_columnForward));
    CUDA_CHECK(cudaFree(d_columnBackward));
    CUDA_CHECK(cudaFree(d_smoothed));

    CUDA_CHECK(cudaEventRecord(totalStop));
    CUDA_CHECK(cudaEventSynchronize(totalStop));
    if (timings != nullptr) {
        CUDA_CHECK(cudaEventElapsedTime(&timings->kernelMs, kernelStart, kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&timings->totalMs, totalStart, totalStop));
    }
    CUDA_CHECK(cudaEventDestroy(totalStart));
    CUDA_CHECK(cudaEventDestroy(totalStop));
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));
}
