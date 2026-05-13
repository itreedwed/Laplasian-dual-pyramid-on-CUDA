cmake_minimum_required(VERSION 3.18)
project(laplacian_filter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# 1. Заголовки OpenCV
set(OPENCV_INC  "C:/opencv/build/include")
# 2. Файл библиотеки OpenCV 
set(OPENCV_LIB  "C:/opencv/build/x64/vc16/lib/opencv_world4120.lib")
# 3. Заголовки CUDA
set(CUDA_INC    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include")
# 4. Библиотека CUDA Runtime
set(CUDA_LIB    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/lib/x64/cudart.lib")

# Создаём цель
add_executable(laplacian_filter
    main.cpp
    filter_cpu.cpp
    filter_gpu.cu
)

# Подключаем заголовки OpenCV и CUDA к main.cpp
target_include_directories(laplacian_filter PRIVATE ${OPENCV_INC} ${CUDA_INC})

# Подключаем библиотеки
target_link_libraries(laplacian_filter PRIVATE "${OPENCV_LIB}" "${CUDA_LIB}")

set_target_properties(laplacian_filter PROPERTIES CUDA_ARCHITECTURES "native")

# Автоматически копируем OpenCV DLL в папку с .exe после сборки
add_custom_command(TARGET laplacian_filter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/x64/vc16/bin/opencv_world4120.dll"
    $<TARGET_FILE_DIR:laplacian_filter>
    COMMENT "Копирование OpenCV DLL в папку сборки..."
)