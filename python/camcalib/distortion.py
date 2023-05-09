import numpy as np
import scipy.optimize as opt
import cv2 as cv

from typing import List, NamedTuple
from .utils import homogenous, hnormalize
from .types import *

__all__ = ["workDistortion"]


def workDistortion(camMat: Matrix3d,
                   distCoeffs: VectorXd,
                   rvecs: List[Vector3d],
                   tvecs: List[Vector3d],
                   objPts: List[List[Vector3d]],
                   imgPts: List[List[Vector2d]],
                   verbose: bool = False):
    params_init = _extractParameters(camMat, distCoeffs, rvecs, tvecs)

    res: opt.OptimizeResult = opt.least_squares(
        _reprojection_error,
        params_init,
        method="lm",
        verbose=verbose,
        args=(objPts, imgPts)
    )

    _composeParameters(res.x, camMat, distCoeffs, rvecs, tvecs)


def _extractParameters(K: Matrix3d,
                       D: VectorXd,
                       rvecs: List[Vector3d],
                       tvecs: List[Vector3d]) -> VectorXd:
    params = []

    # K params: fx, fy, cx, cy
    params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

    # D params: k1, k2, p1, p2
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
    np.copyto(D, params[4:8])

    # compose R, T
    for i in range(len(rvecs)):
        start = 8 + 6 * i
        end = start + 3
        np.copyto(params[start: end], rvecs[i])
        np.copyto(params[start + 3: end + 3], tvecs[i])


def _reprojection_error(params: VectorXd,
                        objPts: List[List[Vector3d]],
                        imgPts: List[List[Vector2d]]):
    # final residuals
    n_imgs = len(objPts)
    n_pts = len(objPts[0])
    residuals = np.zeros([2 * len(objPts) * len(objPts[0])], dtype=np.float64)

    # Get Intrinsic Matrix
    fx, fy, cx, cy = params[0:4]

    # Get Distortion params
    k1, k2, p1, p2 = params[4:8]

    # for all points in the image, add residuals
    for i in range(n_imgs):
        start = 8 + 6 * i
        end = start + 3

        # rotation and translation params
        rvec = params[start: end]
        tvec = params[start + 3: end + 3].reshape(3, 1)
        R = cv.Rodrigues(rvec)[0]

        # calculate residual of each point pair
        for j in range(n_pts):

            # normalized camera point
            norm_pt = (R @ objPts[i][j] + tvec).reshape(-1)
            norm_pt /= norm_pt[2]

            u = norm_pt[0]
            v = norm_pt[1]

            # distortion
            r = np.linalg.norm(norm_pt)
            L = 1. + k1 * pow(r, 2) + k2 * pow(r, 4)
            du = 2. * p1 * u * v + p2 * (r * r + 2. * u * u)
            dv = p1 * (r * r + 2. * v * v) + 2. * p2 * u * v
            u = L * u + du
            v = L * v + dv

            # projection
            pred_xij = fx * u + cx
            pred_yij = fy * v + cy

            # J
            residuals[n_pts * i + 2 * j] = imgPts[i][j][0] - pred_xij
            residuals[n_pts * i + 2 * j + 1] = imgPts[i][j][1] - pred_yij

    return residuals
