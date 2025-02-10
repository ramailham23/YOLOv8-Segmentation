cmake_minimum_required(VERSION 3.8)
project(yolov8_detection C CXX)

set(CMAKE_CXX_STANDARD 17)

# Temukan OpenCV dengan dukungan CUDA
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Temukan CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Temukan YAML-CPP
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Dukungan untuk threading
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Buat executable
add_executable(yo yolov8Normal.cpp inference.cpp)
target_link_libraries(yo
    ${OpenCV_LIBS}
    yaml-cpp
    Threads::Threads
    ${CUDA_LIBRARIES}  # Pastikan CUDA di-link
    opencv_dnn
)
