cmake_minimum_required(VERSION 3.17)

project(camcalib)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})

add_library(camcalib SHARED calibrate.cpp)
target_link_libraries(camcalib Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS})

add_subdirectory(examples)
