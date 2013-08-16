cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/lib/i386-linux-gnu/)
find_package(X11)
find_package(CUDA)
set(LIBS ${LIBS} ${X11_LIBRARIES} ${CUDA_LIBARARIES} m pthread)
add_definitions(-DDEBUG)
set(CUDA_NVCC_FLAGS  -arch=sm_21)
#cuda_add_executable(transpose main.cpp cppIntegration_gold.cpp cppIntegration.cu)
cuda_add_executable(transpose transpose.cu)
target_link_libraries(transpose ${LIBS})


