//
// Created by chris on 2023/5/7.
//
#ifndef CPP_CALIBRATE_H
#define CPP_CALIBRATE_H
#include <vector>
#include <opencv2/core.hpp>

#define INPUT const
#define OUTPUT
#define IN_OUT


void calibrate(INPUT std::vector<std::vector<cv::Point3f>> &objPoints,
               INPUT std::vector<std::vector<cv::Point2f>> &imgPoints,
               INPUT cv::Size imgSize,
               OUTPUT cv::Mat &camMat,
               OUTPUT cv::Mat &distCoeffs,
               OUTPUT cv::Mat &rvecs,
               OUTPUT cv::Mat &tvecs);


#endif //CPP_CALIBRATE_H
