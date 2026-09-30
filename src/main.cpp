#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "filter_cpu.h"
#include "filter_gpu.h"
#include "filter_params.h"

namespace {

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName
              << " [--input <path>] [--output <path>] [--aperture <odd positive K>]\n"
              << "Defaults: --input data/sample.png --output output_laplacian.png --aperture 7\n";
}

int maxAbsoluteDifference(const std::vector<unsigned char>& first,
                          const std::vector<unsigned char>& second) {
    int maximum = 0;
    for (size_t index = 0; index < first.size(); ++index) {
        const int difference = std::abs(static_cast<int>(first[index]) - static_cast<int>(second[index]));
        maximum = std::max(maximum, difference);
    }
    return maximum;
}

} // namespace

int main(int argc, char** argv) {
    std::string inputPath = "data/sample.png";
    std::string outputPath = "output_laplacian.png";
    int apertureK = 7;

    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--help" || option == "-h") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if ((option == "--input" || option == "--output" || option == "--aperture") && argument + 1 >= argc) {
            std::cerr << "Missing value for " << option << std::endl;
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        if (option == "--input") {
            inputPath = argv[++argument];
        } else if (option == "--output") {
            outputPath = argv[++argument];
        } else if (option == "--aperture") {
            try {
                apertureK = std::stoi(argv[++argument]);
            } catch (const std::exception&) {
                std::cerr << "Aperture K must be an integer." << std::endl;
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "Unknown option: " << option << std::endl;
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // Validates K before any work is done.
    const LdpFilterParams params = makeLdpFilterParams(apertureK);

    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: no usable CUDA device found ("
                  << (deviceStatus != cudaSuccess ? cudaGetErrorString(deviceStatus) : "0 devices")
                  << "). An NVIDIA GPU with an up-to-date driver is required." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: unable to load image " << inputPath << std::endl;
        return EXIT_FAILURE;
    }

    const int width = image.cols;
    const int height = image.rows;
    const int channels = image.channels();
    std::vector<unsigned char> input(image.total() * channels);
    std::vector<unsigned char> cpuOutput(input.size());
    std::vector<unsigned char> gpuOutput(input.size());
    std::memcpy(input.data(), image.data, input.size());

    std::cout << "Image: " << width << "x" << height << ", channels: " << channels << "\n"
              << "LDP filter parameters: K = " << apertureK
              << ", alpha = " << params.alpha
              << ", beta = " << params.beta << std::endl;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    applyLaplacianCPU(input.data(), cpuOutput.data(), width, height, channels, apertureK);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // Warm-up run: CUDA context creation and first-launch overhead are excluded from timing.
    applyLaplacianGPU(input.data(), gpuOutput.data(), width, height, channels, apertureK);

    GpuTimings gpuTimings;
    applyLaplacianGPU(input.data(), gpuOutput.data(), width, height, channels, apertureK, &gpuTimings);

    const double megapixels = static_cast<double>(width) * height * channels / 1e6;
    std::cout << "\n=== RESULTS ===\n";
    std::cout << "CPU/OpenMP time:              " << cpuTimeMs << " ms\n";
    std::cout << "GPU time (with transfers):    " << gpuTimings.totalMs << " ms\n";
    std::cout << "GPU time (kernels only):      " << gpuTimings.kernelMs << " ms\n";
    std::cout << "Speedup (with transfers):     " << cpuTimeMs / gpuTimings.totalMs << "x\n";
    std::cout << "Speedup (kernels only):       " << cpuTimeMs / gpuTimings.kernelMs << "x\n";
    std::cout << "GPU kernel throughput:        "
              << megapixels / (gpuTimings.kernelMs / 1000.0) << " MPix/s\n";
    std::cout << "Maximum CPU/GPU absolute difference: "
              << maxAbsoluteDifference(cpuOutput, gpuOutput) << "\n";

    cv::Mat output(height, width, image.type());
    std::memcpy(output.data, gpuOutput.data(), gpuOutput.size());
    if (!cv::imwrite(outputPath, output)) {
        std::cerr << "Error: unable to save image " << outputPath << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Filtered image saved to: " << outputPath << std::endl;

    return EXIT_SUCCESS;
}
