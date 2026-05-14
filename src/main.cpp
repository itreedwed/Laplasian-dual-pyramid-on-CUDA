#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include "filter_cpu.h"
#include "filter_gpu.h"

void printHelp(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
        << "  --input <path>   Path to input image (default: input.png)\n"
        << "  --output <path>  Path to output image (default: output_laplacian.png)\n"
        << "  --help           Show this help message\n\n"
        << "Example:\n  " << progName << " --input data/input.png --output result.png\n";
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8); // Fix Windows console encoding

    std::string inputPath = "input.png";
    std::string outputPath = "output_laplacian.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
        else if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\nUse --help for more information.\n";
            return 1;
        }
    }

    if (!std::filesystem::exists(inputPath)) {
        std::cerr << "Error: file '" << inputPath << "' not found.\n";
        return 1;
    }

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: failed to read image. Check format (supports PNG, JPG, BMP).\n";
        return 1;
    }

    int W = img.cols, H = img.rows, C = img.channels();
    std::vector<unsigned char> input(H * W * C);
    std::vector<unsigned char> outCPU(H * W * C);
    std::vector<unsigned char> outGPU(H * W * C);

    std::memcpy(input.data(), img.data, input.size());

    const float kernel[49] = {
            -1, -2, -3, -3, -3, -2, -1,
            -2, -4, -6, -6 ,-6 - 4 ,-2,
            -3, -6, 5, 19, 5, -6, -3,
            -3, -6, 19, 48, 19, -6, -3,
            -3, -6, 5, 19, 5, -6, -3,
            -2, -4, -6, -6, -6, -4, -2,
            -1, -2, -3, -3, -3, -2, -1,
    };

    std::cout << "Image: " << W << "x" << H << ", Channels: " << C << "\n";

    auto startCPU = std::chrono::high_resolution_clock::now();
    applyLaplacianCPU(input.data(), outCPU.data(), W, H, C, kernel, 7);
    auto endCPU = std::chrono::high_resolution_clock::now();
    double timeCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

    applyLaplacianGPU(input.data(), outGPU.data(), W, H, C, kernel, 7); // Warmup

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    applyLaplacianGPU(input.data(), outGPU.data(), W, H, C, kernel, 7);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeGPU = 0;
    cudaEventElapsedTime(&timeGPU, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    double speedup = timeCPU / timeGPU;
    double mpixels = (static_cast<double>(W) * H * C) / 1e6;

    std::cout << "\nResults:\n";
    std::cout << "  CPU: " << timeCPU << " ms\n";
    std::cout << "  GPU: " << timeGPU << " ms\n";
    std::cout << "  Speedup: " << speedup << "x\n";
    std::cout << "  GPU Throughput: " << (mpixels / (timeGPU / 1000.0)) << " Mpix/s\n";

    cv::Mat outMat(H, W, img.type());
    std::memcpy(outMat.data, outGPU.data(), outGPU.size());

    if (!cv::imwrite(outputPath, outMat)) {
        std::cerr << "Error: failed to save result to '" << outputPath << "'\n";
        return 1;
    }
    std::cout << "Result saved: " << outputPath << "\n";

    return 0;
}