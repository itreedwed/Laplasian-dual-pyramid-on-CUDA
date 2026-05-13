# Laplacian Double Pyramid Filter (7×7) with CUDA Acceleration

Implementation of edge detection filter (Laplacian operator with 7×7 aperture) with GPU parallel acceleration via CUDA. The project includes CPU vs GPU performance comparison, automatic timing benchmarks, and throughput measurements.

## 📊 What it does
* Loads an image using OpenCV.
* Applies 7×7 convolution (Laplacian kernel) sequentially on CPU.
* Performs the same operation in parallel on GPU via CUDA kernel.
* Measures execution time (std::chrono for CPU, cudaEvent for GPU).
* Outputs comparison metrics and saves the result with detected edges.

## ⚡ Performance Results (*)
| Resolution | CPU (ms) | GPU (ms) |  Speedup  |
|------------|----------|----------|-----------|
|  640×480   | ~48      | ~1,2     | **~40×**  |
| 1280×720   | ~164     | ~3,1     | **~51×**  |
| 1920×1080  | ~439     | ~6,3     | **~69×**  |
| 3000×2000  | ~1402    | ~12,3    | **~113×** |
| 6000×4160  | ~3663    | ~42,9    | **~99×**  |

## 🛠 Requirements
- Windows 10/11
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (v11.8+)
- [OpenCV](https://opencv.org/releases/) (v4.x, prebuilt Windows)
- CMake 4.32+
- MSVC (Visual Studio 2022) / Ninja

## 📦 Building
```bash
git clone https://github.com/itreedwed/laplacian-cuda-filter.git
cd laplacian-cuda-filter
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
## ⚠️ If CUDA cannot find OpenCV/CUDA, specify paths explicitly:
```
cmake .. -DOpenCV_DIR="C:/opencv/build" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
```

## 🚀 Usage
```
# Run with default files (input.png / output_laplacian.png)
./laplacian_filter.exe

# Specify custom paths
./laplacian_filter.exe --input data/city.png --output edges.png

# Show help
./laplacian_filter.exe --help
```
The *input.png* file should be in the working directory, or provide an absolute path.

### 🎓 This code was developed as part of the task for group project-based learning. The source code is open and can be used to learn


(*) PC: AMD Ryzen 5 2600, 16GB DDR4, NVidia GeForce GTX 1660 Super, Win 11 25H2.