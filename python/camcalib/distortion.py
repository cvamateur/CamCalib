import numpy as np
import scipy.optimize as opt
import cv2 as cv

from typing import List, NamedTuple
from .utils import homogenous, hnormalize
from .types import *

__all__ = ["workDistortion"]


class CalibrationResult(NamedTuple):
    cameraMatrix: Matrix3d  # Camera intrinsic matrix.
    distCoeffs: VectorXd  # Distortion parameters
    rvecs: List[Vector3d]  # Rotation vectors of each image.
    tvecs: List[Vector3d]  # Translation vector of each image.


def workDistortion(camMat: Matrix3d,
                   distCoeffs: VectorXd,
                   rvecs: List[Vector3d],
                   tvecs: List[Vector3d],
                   objPts: List[List[Vector3d]],
                   imgPts: List[List[Vector2d]]):
    params = _extractParameters(camMat, distCoeffs, rvecs, tvecs)

    res = opt.least_squares(
        _reprojection_error,

    )

    return CalibrationResult(camMat, distCoeffs, rvecs, tvecs)


def _extractParameters(K: Matrix3d,
                       D: VectorXd,
                       rvecs: List[Vector3d],
                       tvecs: List[Vector3d]) -> VectorXd:
    params = []

    # K params: fx, fy, cx, cy
    params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

    # D params: k1, k2, p1, p2 [,k3]
    params.extend(D)

    # R, T params: r1, r2, r3, tx, ty, tz
    for i in range(len(rvecs)):
        params.extend(rvecs[i])
        params.extend(tvecs[i])

    return np.array(params, dtype=np.float64)


def _composeParameters(params: VectorXd,
                       K: Matrix3d,
                       D: VectorXd,
                       rvecs: List[Vector3d],
                       tvecs: List[Vector3d]) -> None:
    # compose K
    fx, fy, cx, cy = params[0:4]
    K_ = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float64)
    np.copyto(K, K_.reshape(3, 3))

    # compose D
    np.copyto(D, params[4:4 + len(D)])

    # compose R, T
    _off = 4 + len(D)
    for i in range(len(rvecs)):
        start = _off + 6 * i
        end = start + 3
        np.copyto(params[start: end], rvecs[i])
        np.copyto(params[start + 3: end + 3], tvecs[i])


def _reprojection_error(params: VectorXd,
                        D_dim: int,
                        objPts: List[List[Vector3d]],
                        imgPts: List[List[Vector2d]]):
    # final residuals
    n_imgs = len(objPts)
    n_pts = len(objPts[0])
    residuals = np.zeros([2 * len(objPts) * len(objPts[0])], dtype=np.float64)

    # Get Intrinsic Matrix
    fx, fy, cx, cy = params[0:4]
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float64)
    K = K.reshape(3, 3)

    # Get Distortion params
    D = params[4:4 + D_dim]

    # for all points in the image, add residuals
    _off = 4 + D_dim
    for i in range(n_imgs):
        start = _off + 6 * i
        end = start + 3

        # rotation and translation params
        rvec = params[start: end]
        tvec = params[start + 3: end + 3].reshape(3, 1)
        R = cv.Rodrigues(rvec)[0]

        # Extrinsic of this image
        Ex = np.concatenate([R, tvec], axis=1)

        # calculate residual of each point pair
        for j in range(n_pts):
            Pw = homogenous(objPts[i][j])

            # normalized camera point
            norm_Px = Ex @ Pw
            norm_Px /= norm_Px[2]