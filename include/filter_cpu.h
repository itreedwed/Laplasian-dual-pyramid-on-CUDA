#pragma once

void applyLaplacianCPU(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels,
                       const float* kernel, int kSize);