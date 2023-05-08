//
// Created by chris on 2023/5/7.
//
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <opencv2/core/eigen.hpp>

#include "calibrate.h"


static Eigen::VectorXd _v(const Eigen::Matrix3d &H, int i, int j) {
    Eigen::VectorXd v(6);
    v[0] = H(0, i) * H(0, j);
    v[1] = H(1, i) * H(0, j) + H(0, i) * H(1, j);
    v[2] = H(2, i) * H(0, j) + H(0, i) * H(2, j);
    v[3] = H(1, i) * H(1, j);
    v[4] = H(2, i) * H(1, j) + H(1, i) * H(2, j);
    v[5] = H(2, i) * H(2, j);
    return v;
}


void calibrate(INPUT std::vector<std::vector<cv::Point3f>> &objPoints,
               INPUT std::vector<std::vector<cv::Point2f>> &imgPoints,
               INPUT cv::Size imgSize,
               OUTPUT cv::Mat &camMat,
               OUTPUT cv::Mat &distCoeffs,
               OUTPUT cv::Mat &rvecs,
               OUTPUT cv::Mat &tvecs) {

    assert(objPoints.size() == imgPoints.size() && objPoints.size() >= 3);
    const int num_images = objPoints.size();
    const int num_points = objPoints[0].size();


    std::vector<Eigen::Matrix3d> H(num_images);
    Eigen::MatrixXd A(2 * num_points, 8);
    Eigen::VectorXd b(2 * num_points);
    Eigen::BDCSVD<Eigen::MatrixXd> svd;

    Eigen::MatrixXd V(2 * num_images, 6);

    for (int i = 0; i < num_images; ++i) {
        const auto &objPts_i = objPoints[i];
        const auto &imgPts_i = imgPoints[i];
        for (int j = 0; j < num_points; ++j) {
            const auto &objPt = objPts_i[j];
            const auto &imgPt = imgPts_i[j];
            Eigen::Vector3d Pi = Eigen::Vector3d(objPt.x, objPt.y, 1);

            A.block<1, 3>(2 * j, 0) = Pi;
            A.block<1, 3>(2 * j, 3) = Eigen::Vector3d::Zero();
            A.block<1, 2>(2 * j, 6) = Eigen::Vector2d(-imgPt.x * Pi[0], -imgPt.x * Pi[1]);
            b[2 * j] = imgPt.x;

            A.block<1, 3>(2 * j + 1, 0) = Eigen::Vector3d::Zero();
            A.block<1, 3>(2 * j + 1, 3) = Pi;
            A.block<1, 2>(2 * j + 1, 6) = Eigen::Vector2d(-imgPt.y * Pi[0], -imgPt.y * Pi[1]);
            b[2 * j + 1] = imgPt.y;
        }

        svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd S_inv(8, 2 * num_points);
        S_inv.setZero();
        for (int k = 0; k < 8; ++k)
            S_inv(k, k) = ( s[k] < 1e-6 ? 0.0 : 1. / s[k] );

        Eigen::VectorXd Hi = svd.matrixV() * S_inv * svd.matrixU().transpose() * b;
        Eigen::VectorXd Hi_homo = Hi.homogeneous();
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Hi_mat(Hi_homo.data(), 3, 3);
        H[i] = Hi_mat;

        V.row(2 * i) = _v(Hi_mat, 0, 1);
        V.row(2 * i + 1) = _v(Hi_mat, 0, 0) - _v(Hi_mat, 1, 1);
    }

    // Solve B
    svd.compute(V, Eigen::ComputeThinV);
    Eigen::VectorXd b_ = svd.matrixV().col(5);
    Eigen::Matrix3d B;
    B << b_[0], b_[1], b_[2],
            b_[1], b_[3], b_[4],
            b_[2], b_[4], b_[5];

    // decompose B to K
    Eigen::LLT<Eigen::Matrix3d, Eigen::Lower> llt(B);
    Eigen::Matrix3d L = llt.matrixL();

    Eigen::Matrix3d K = L.transpose().inverse();
    K /= K(2, 2);
    cv::eigen2cv(K, camMat);

    // Extrinsic
    rvecs.create(num_images, 3, CV_64F);
    tvecs.create(num_images, 3, CV_64F);

    Eigen::Matrix3d K_inv = K.inverse();
    Eigen::Vector3d h1, h2, h3;
    Eigen::Vector3d rvec, tvec;
    Eigen::Matrix3d R;
    Eigen::AngleAxisd rotA;

    for (int i = 0; i < H.size(); ++i) {
        const auto &Hi = H[i];
        h1 = Hi.col(0);
        h2 = Hi.col(1);
        h3 = Hi.col(2);

        const double rho = 1.0 / ( K_inv * h1 ).norm();
        R.col(0) = rho * K_inv * h1;
        R.col(1) = rho * K_inv * h2;
        R.col(2) = R.col(0).cross(R.col(1));
        rotA.fromRotationMatrix(R); // Rodrigues rotation vector
        rvec = rotA.angle() * rotA.axis();
        tvec = rho * K_inv * h3;

        rvecs.at<double>(i, 0) = rvec[0];
        rvecs.at<double>(i, 1) = rvec[1];
        rvecs.at<double>(i, 2) = rvec[2];
        tvecs.at<double>(i, 0) = tvec[0];
        tvecs.at<double>(i, 1) = tvec[1];
        tvecs.at<double>(i, 2) = tvec[2];
    }
}