# Laplacian Double Pyramid (LDP) Filter on CUDA

[![Build](https://github.com/itreedwed/Laplasian-dual-pyramid-on-CUDA/actions/workflows/build.yml/badge.svg)](https://github.com/itreedwed/Laplasian-dual-pyramid-on-CUDA/actions/workflows/build.yml)

A two-dimensional **recursive separable** Laplacian filter for colour images,
accelerated with CUDA. The program runs the same filter on the CPU (OpenMP) and
on the GPU, checks that both results match, and reports timings and speedup.

![Input and filtered output](docs/example.png)

## How it works

For an odd positive aperture `K` the recursive coefficients are

```
alpha = 2 / (K + 1)
beta  = 1 - alpha
```

1. Every image row is filtered with the first-order recursive filter
   `y[n] = alpha * x[n] + beta * y[n-1]` in the forward and in the backward
   direction; the two results are averaged (symmetric, zero-phase low-pass).
2. The same forward/backward pass is applied to every column.
3. The Laplacian response is `input - lowPass + 128`, saturated to 8 bits.

Larger `K` means stronger smoothing and therefore thicker, lower-frequency
edges in the output. `K = 1` gives a uniform grey (128) image.

On the GPU each CUDA thread processes one row or one column of one colour
channel; `alpha` and `beta` are stored in constant memory.

## Performance

2560×1600 RGB image, `K = 7`, Release build:

| Implementation | Time | Speedup vs CPU |
|---|---|---|
| CPU, OpenMP (12 threads) | ~345 ms | 1× |
| GPU, including allocation and host↔device copies | ~23 ms | ~15× |
| GPU, kernels only | ~4.6 ms | ~75× |

PC: AMD Ryzen 5 2600, 16 GB DDR4, NVIDIA GeForce GTX 1660 SUPER, Windows 11.
The CPU and GPU outputs are bit-identical (maximum difference 0).

## Quick start: prebuilt Windows binary

1. Download `laplacian_filter-<version>-Windows-x64.zip` from
   [Releases](https://github.com/itreedwed/Laplasian-dual-pyramid-on-CUDA/releases)
   and unpack it.
2. Run `laplacian_filter.exe` from the unpacked folder. It filters
   `data/sample.png` and writes `output_laplacian.png`.

Requirements: 64-bit Windows 10/11, an NVIDIA GPU of the Turing generation or
newer (GTX 16xx, RTX 20xx and later), an NVIDIA driver version 580 or newer and the
[Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
The CUDA Toolkit is **not** needed to run the binary.

## Building from source

### Requirements

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x
- CMake 3.25 or newer and Ninja
- A C++17 compiler supported by your CUDA version:
  Visual Studio 2022/2026 with the "Desktop development with C++" workload on
  Windows, GCC on Linux
- OpenCV 4 — installed automatically by vcpkg, or installed by you

### Windows, OpenCV installed automatically (recommended)

Open **x64 Native Tools Command Prompt** (or Developer PowerShell) for Visual
Studio. Visual Studio ships vcpkg and sets `VCPKG_ROOT` there; if you use a
standalone vcpkg, set `VCPKG_ROOT` to its folder.

```bat
git clone https://github.com/itreedwed/Laplasian-dual-pyramid-on-CUDA.git
cd Laplasian-dual-pyramid-on-CUDA
cmake --preset release
cmake --build --preset release
build\release\laplacian_filter.exe
```

The first configure downloads and builds OpenCV (it takes a while, once).
Visual Studio and VS Code (CMake Tools) also pick up the presets from
`CMakePresets.json`: just open the folder and choose the `release` preset.

### Windows, with a prebuilt OpenCV

Download OpenCV for Windows from [opencv.org](https://opencv.org/releases/),
unpack it (for example to `C:\opencv`) and pass the folder that contains
`OpenCVConfig.cmake`:

```bat
cmake --preset release-system -DOpenCV_DIR=C:/opencv/build/x64/vc16/lib
cmake --build --preset release-system
build\release-system\laplacian_filter.exe
```

The OpenCV DLL is copied next to the executable automatically.

### Linux

```bash
sudo apt install libopencv-dev ninja-build cmake   # plus the CUDA Toolkit
cmake --preset release-system
cmake --build --preset release-system
./build/release-system/laplacian_filter
```

The Linux build is checked on every push by GitHub Actions.

### Target GPU

By default the code is compiled for the GPU installed in the building machine
(`CMAKE_CUDA_ARCHITECTURES=native`). To build for another GPU, pass its
compute capability, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`.

## Usage

```
laplacian_filter [--input <path>] [--output <path>] [--aperture <odd positive K>]
```

| Option | Default | Meaning |
|---|---|---|
| `--input` | `data/sample.png` | Input image (any format OpenCV can read) |
| `--output` | `output_laplacian.png` | Output image |
| `--aperture` | `7` | Odd positive aperture `K`: 1, 3, 5, 7, 9, 15, 31, … |

Example output:

```
Image: 1280x800, channels: 3
LDP filter parameters: K = 7, alpha = 0.25, beta = 0.75

=== RESULTS ===
CPU/OpenMP time:              63.9 ms
GPU time (with transfers):    8.24 ms
GPU time (kernels only):      1.54 ms
Speedup (with transfers):     7.75x
Speedup (kernels only):       41.5x
GPU kernel throughput:        1998 MPix/s
Maximum CPU/GPU absolute difference: 0
Filtered image saved to: output_laplacian.png
```

## Making a release package (for maintainers)

From an x64 Native Tools Command Prompt:

```bat
cmake --preset release-dist -DOpenCV_DIR=C:/opencv/build/x64/vc16/lib -DOpenCV_LICENSE_FILE=C:/opencv/LICENSE.txt
cmake --build --preset release-dist
cpack --preset release-dist
```

This builds for every GPU generation supported by CUDA 13 (plus PTX for future
GPUs), links the CUDA runtime statically and packs the executable, the OpenCV
DLL, licences and the sample image into
`build/release-dist/laplacian_filter-<version>-Windows-x64.zip`.

## Project structure

```
include/          headers
src/main.cpp      command line, timing, CPU/GPU comparison
src/filter_cpu.cpp  reference implementation (OpenMP)
src/filter_gpu.cu   CUDA kernels
src/filter_params.cpp  alpha/beta from the aperture K
data/sample.png   test image
CMakePresets.json build presets
vcpkg.json        OpenCV dependency for vcpkg
```

---

This code was developed as part of a group project-based learning task. It is
open source under the MIT licence and can be used for learning.
