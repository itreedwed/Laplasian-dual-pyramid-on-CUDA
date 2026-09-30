cmake_minimum_required(VERSION 3.21)
project(laplacian_filter VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Match OpenCV's Debug/Release ABI on MSVC.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# CUDA Toolkit is located through the active compiler/toolkit installation.
# OpenCV is found either through vcpkg (see vcpkg.json and CMakePresets.json)
# or through a prebuilt package:
#   cmake -S . -B build -DOpenCV_DIR=<directory-containing-OpenCVConfig.cmake>
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# "native" builds only for the GPU of the current machine. Release packages
# override this with a list of architectures (see the "release-dist" preset).
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures to build for; override for a specific target GPU")
endif()

add_executable(laplacian_filter
    src/main.cpp
    src/filter_cpu.cpp
    src/filter_params.cpp
    src/filter_gpu.cu
)

target_include_directories(laplacian_filter PRIVATE include)

# The CUDA runtime is linked statically, so users only need the NVIDIA driver.
target_link_libraries(laplacian_filter PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart_static
    OpenMP::OpenMP_CXX
)

if (WIN32)
    # Copy the OpenCV runtime DLL(s) next to the executable so it runs from the build directory.
    add_custom_command(TARGET laplacian_filter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:laplacian_filter>
            $<TARGET_FILE_DIR:laplacian_filter>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs..."
    )
endif()

# `cmake --install` / `cpack` produce a self-contained folder or ZIP archive.
install(TARGETS laplacian_filter RUNTIME DESTINATION .)
if (WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:laplacian_filter> DESTINATION .)
endif()
install(FILES README.md LICENSE.txt DESTINATION .)
install(FILES data/sample.png DESTINATION data)
if (DEFINED OpenCV_LICENSE_FILE AND EXISTS "${OpenCV_LICENSE_FILE}")
    install(FILES "${OpenCV_LICENSE_FILE}" DESTINATION . RENAME LICENSE-OpenCV.txt)
endif()

set(CPACK_GENERATOR ZIP)
set(CPACK_PACKAGE_NAME laplacian_filter)
set(CPACK_PACKAGE_FILE_NAME
    "laplacian_filter-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-x64")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
include(CPack)
