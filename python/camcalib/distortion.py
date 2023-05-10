import math
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
        _jacobian,
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
        params.extend(rvecs[i].flatten())
        params.extend(tvecs[i].flatten())

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
    np.copyto(D, params[4:9])

    # compose R, T
    for i in range(len(rvecs)):
        start = 9 + 6 * i
        end = start + 3
        np.copyto(params[start: end, None], rvecs[i])
        np.copyto(params[start + 3: end + 3, None], tvecs[i])


class _Distortion:

    def __init__(self, k1, k2, p1, p2, k3):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3

    def x(self, x, y, r):
        L = 1. + sum(k * pow(r, 2 * p) for p, k in enumerate([self.k1, self.k2, self.k3], start=1))
        qx = 2. * self.p1 * x * y + self.p2 * (r * r + 2. * x * x)
        return L * x + qx

    def y(self, y, x, r):
        L = 1. + sum(k * pow(r, 2 * p) for p, k in enumerate([self.k1, self.k2, self.k3], start=1))
        qy = self.p1 * (r * r + 2. * y * y) + 2. * self.p2 * x * y
        return L * y * qy


def _reprojection_error(params: VectorXd,
                        objPts: List[List[Vector3d]],
                        imgPts: List[List[Vector2d]]):
    # final residuals
    n_imgs = len(objPts)
    n_pts = len(objPts[0])
    residuals = np.zeros([len(objPts) * len(objPts[0]) * 2], dtype=np.float64)

    # Get Intrinsic Matrix
    fx, fy, cx, cy = params[0:4]

    # Get Distortion params
    k1, k2, p1, p2, k3 = params[4:9]
    distort = _Distortion(k1, k2, p1, p2, k3)

    # for all points in the image, add residuals
    for i in range(n_imgs):
        start = 9 + 6 * i
        end = start + 3

        # rotation and translation params
        rvec = params[start: end].reshape(3, 1)
        tvec = params[start + 3: end + 3].reshape(3, 1)
        R = cv.Rodrigues(rvec)[0]

        # calculate residual of each point pair
        for j in range(n_pts):
            Pw_ij = objPts[i][j].reshape(3, 1)

            # normalized camera point
            Pc = (R @ Pw_ij + tvec).reshape(-1)
            Xc = Pc[0]
            Yc = Pc[1]
            Zc = Pc[2]
            u = Xc / Zc
            v = Yc / Zc

            # distortion
            r = math.sqrt(u * u + v * v)
            up = distort.x(u, v, r)
            vp = distort.y(u, v, r)

            # projection
            pred_xij = fx * up + cx
            pred_yij = fy * vp + cy

            # J
            residuals[2 * n_pts * i + 2 * j + 0] = imgPts[i][j][0] - pred_xij
            residuals[2 * n_pts * i + 2 * j + 1] = imgPts[i][j][1] - pred_yij

    return residuals


def _jacobian(params: VectorXd,
              objPts: List[List[Vector3d]],
              imgPts: List[List[Vector2d]]):
    n_imgs = len(objPts)
    n_pts = len(objPts[0])
    jac = np.zeros([n_imgs * n_pts * 2, params.size], dtype=np.float64)

    # Get Intrinsic Matrix
    fx, fy, cx, cy = params[0:4]

    # Get Distortion params
    k1, k2, p1, p2, k3 = params[4:9]
    distort = _Distortion(k1, k2, p1, p2, k3)

    for i in range(n_imgs):
        start = 9 + 6 * i
        end = start + 3

        # rotation and translation params
        rvec = params[start: end].reshape(3, 1)
        tvec = params[start + 3: end + 3].reshape(3, 1)
        R, dRdr = cv.Rodrigues(rvec)

        for j in range(n_pts):
            Pw = objPts[i][j].reshape(3, 1)

            Pc = (R @ Pw + tvec).reshape(-1)
            Xc = Pc[0]
            Yc = Pc[1]
            Zc = Pc[2]
            u = Xc / Zc
            v = Yc / Zc

            r = math.sqrt(u * u + v * v)
            up = distort.x(u, v, r)
            vp = distort.y(u, v, r)

            # ∂Jx
            dJx = jac[2 * n_pts * i + 2 * j]

            # gradient on K
            dJx[0] = -up  # dfx
            dJx[1] = 0  # dfy
            dJx[2] = -1  # dcx
            dJx[3] = 0  # dcy

            # gradient on distortions
            dJx[4] = -fx * u * r * r  # dk1
            dJx[5] = -fx * u * pow(r, 4)  # dk2
            dJx[6] = -fx * 2. * u * v  # dp1
            dJx[7] = -fx * (r * r + 2. * u * u)  # dp2
            dJx[8] = -fx * u * pow(r, 6)  # dk3

            # gradient on rvec and tvec
            dupdu = opt.approx_fprime(u, distort.x, None, v, r)[0]
            dJxdu = -fx * dupdu
            dJxdXc = dJxdu / Zc
            dJxdZc = dJxdu * (-Xc / Zc ** 2)

            dJxdPc = np.array([dJxdXc, 0., dJxdZc], dtype=np.float64).reshape(3, 1)
            dJxdR = dJxdPc @ Pw.T
            dJxdr = dJxdR.reshape(1, 9) @ dRdr.T
            dJxdt = dJxdPc
            dJx[start:end] = dJxdr.flatten()
            dJx[start + 3:end + 3] = dJxdt.flatten()

            # ∂Jy
            dJy = jac[2 * n_pts * i + 2 * j + 1]

            # gradient on K
            dJy[0] = 0
            dJy[1] = -vp
            dJy[2] = 0
            dJy[3] = -1

            # gradient on distortions
            dJy[4] = -fy * v * r * r
            dJy[5] = -fy * v * pow(r, 4)
            dJy[6] = -fy * (r * r + 2. * v * v)
            dJy[7] = -fy * 2. * u * v
            dJy[8] = -fy * v * pow(r, 6)

            dvpdv = opt.approx_fprime(v, distort.y, None, u, r)[0]
            dJydv = -fy * dvpdv
            dJydYc = dJydv / Zc
            dJydZc = dJydv * (-Yc / Zc ** 2)

            dJydPc = np.array([0., dJydYc, dJydZc], dtype=np.float64).reshape(3, 1)
            dJydR = dJydPc @ Pw.T
            dJydr = dJydR.reshape(1, 9) @ dRdr.T
            dJydt = dJydPc
            dJy[start:end] = dJydr.flatten()
            dJy[start + 3: end + 3] = dJydt.flatten()

    return jac
