//
// Created by chris on 2023/5/7.
//
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../calibrate.h"


int CHESSBOARD[2];  // e.g  11x7


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <PathPattern> <ChessboardWidth> <ChessboardHeight>" << std::endl;
        std::cerr << "example: " << argv[0] << " ./imgs/*.jpg 11 7" << std::endl;
        exit(-1);
    }
    CHESSBOARD[0] = atoi(argv[2]);
    CHESSBOARD[1] = atoi(argv[3]);

    // Create a vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f>> objpoints;

    // Create a vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Define the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHESSBOARD[1]; ++i) {
        for (int j = 0; j < CHESSBOARD[0]; ++j) {
            objp.emplace_back(j, i, 0);
        }
    }

    // Extract path of individual image stored in a given directory
    std::vector<cv::String> paths;
    cv::glob(argv[1], paths);
    cv::Mat img_bgr, img_gray;


    // Vector to store the pixel coordinates of detected chessboard corners
    std::vector<cv::Point2f> corner_pts;
    bool success;
    for (const auto & path : paths) {
        img_bgr = cv::imread(path, cv::IMREAD_COLOR);
        cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);

        // Find corners
        // If desired number of corners are found in the image then success = true
        success = cv::findChessboardCorners(img_gray,
                                            cv::Size(CHESSBOARD[0], CHESSBOARD[1]),
                                            corner_pts,
                                            cv::CALIB_CB_ADAPTIVE_THRESH |
                                            cv::CALIB_CB_FAST_CHECK |
                                            cv::CALIB_CB_NORMALIZE_IMAGE);

        // If desired number of corners are detected, we refine
        // the pixel coordinates and display them on the images
        // of checkerboard
        if (success) {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // refine pixel coordinates for given 2d points
            cv::cornerSubPix(img_gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Display the detected corner points on chessboard
            cv::drawChessboardCorners(img_bgr, cv::Size(CHESSBOARD[0], CHESSBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Corners", img_bgr);
        cv::waitKey(100);
    }
    cv::destroyAllWindows();

    // Perform camera calibration by passing the value of known 3D points (objpoints)
    // and corresponding pixel coordinates of the
    // detected corners (imgpoints)
    cv::Mat camMat, distCoeffs, rvecs, tvecs;
    calibrate(objpoints, imgpoints, img_gray.size(), camMat, distCoeffs, rvecs, tvecs);

    std::cout << "Intrinsic Matrix:\n" << camMat << std::endl;
    std::cout << "Lens Distortion:\n" << distCoeffs << std::endl;
    std::cout << "Extrinsic Rotation:\n" << rvecs << std::endl;
    std::cout << "Extrinsic Translation:\n" << tvecs << std::endl;

    /*
    // Trying to undistort the image using the camera parameters obtained from calibration
    img_bgr = cv::imread(paths[0], cv::IMREAD_COLOR);
    cv::Mat dst1, dst2, mapx, mapy, newCamMat;
    cv::Size imgsz = img_bgr.size();

    // Refining the camera matrix using parameters obtained by calibration
    newCamMat = cv::getOptimalNewCameraMatrix(camMat, distCoeffs, imgsz, 1, imgsz, 0);

    // Method 1
    cv::undistort(img_bgr, dst1, camMat, distCoeffs, newCamMat);

    // Method 2
    cv::initUndistortRectifyMap(camMat, distCoeffs, cv::Mat(), newCamMat, imgsz, CV_32F, mapx, mapy);
    cv::remap(img_bgr, dst2, mapx, mapy, cv::INTER_LINEAR);

    cv::imshow("Origin", img_bgr);
    cv::imshow("Method1", dst1);
    cv::imshow("Method2", dst2);
    cv::waitKey(0);
    cv::destroyAllWindows();
    */

    return 0;
}


