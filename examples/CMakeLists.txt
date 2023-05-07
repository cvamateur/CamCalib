project(calibExamples)

add_executable(calib_cv calib_cv.cpp)
target_include_directories(calib_cv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calib_cv ${OpenCV_LIBS})

add_executable(calib calib.cpp)
target_include_directories(calib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calib PRIVATE Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS} camcalib)