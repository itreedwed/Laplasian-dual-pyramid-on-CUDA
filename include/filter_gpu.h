#pragma once

void applyLaplacianGPU(const unsigned char* h_input, unsigned char* h_output,
                       int width, int height, int channels,
                       const float* h_kernel, int kSize);