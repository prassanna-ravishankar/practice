cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/lib/i386-linux-gnu/)
find_package(X11)
find_package(CUDA)
set(LIBS ${LIBS} ${X11_LIBRARIES} ${CUDA_LIBARARIES} m pthread)
add_definitions(-DDEBUG)
set(CUDA_NVCC_FLAGS  -arch=sm_21)
set(CUDA_FILES kernel.cu)
include_directories(/home/masa/project/programming/gpu/NVIDIA_CUDA-5.5_Samples/common/inc /home/masa/project/dev/practice/cuda/cppIntegration)
cuda_add_executable(cuda main.cpp ${CUDA_FILES})


