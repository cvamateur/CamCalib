cmake_minimum_required(VERSION 3.17)

project(camcalib)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)


include_directories("./include" ${OpenCV_INCLUDE_DIRS})

add_library(camcalib SHARED src/calibrate.cpp)
target_link_libraries(camcalib Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS})

add_subdirectory(examples)
