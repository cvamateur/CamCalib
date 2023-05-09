import math
import numpy as np
import scipy.optimize as opt

from typing import List
from .utils import homogenous
from .types import *

__all__ = ["findHomography"]


def findHomography(objPts: List[Vector3d], imgPts: List[Vector2d]) -> Matrix3d:
    assert len(objPts) == len(imgPts) and len(objPts) >= 4
    objNormMat = _getNormalizationMatrix(objPts)
    imgNormMat = _getNormalizationMatrix(imgPts)

    n_pts = len(objPts)
    A: MatrixXd = np.zeros([2 * n_pts, 8])
    b: VectorXd = np.zeros([2 * n_pts, 1])
    for i in range(n_pts):
        objPt: Vector3d = homogenous(objPts[i][:2])
        imgPt: Vector3d = homogenous(imgPts[i])

        objPt = objNormMat @ objPt
        imgPt = imgNormMat @ imgPt

        A[2 * i, 0:3, None] = objPt
        A[2 * i, 3:6, None] = np.zeros([3, 1])
        A[2 * i, 6:8, None] = -imgPt[0] * objPt[0:2]
        b[2 * i, 0] = imgPt[0]

        A[2 * i + 1, 0:3, None] = np.zeros([3, 1])
        A[2 * i + 1, 3:6, None] = objPt
        A[2 * i + 1, 6:8, None] = -imgPt[1] * objPt[0:2]
        b[2 * i + 1, 0] = imgPt[1]

    H: Matrix3d
    U: MatrixXd
    D: MatrixXd
    Vt: MatrixXd
    if n_pts == 4:
        H_vec = np.linalg.inv(A) @ b
    else:
        U, D, Vt = np.linalg.svd(A)
        D_inv = np.zeros([8, 2 * n_pts], dtype=np.float64)
        for k in range(8):
            D_inv[k, k] = 0.0 if D[k] < 1e-6 else 1. / D[k]
        H_vec = Vt.T @ D_inv @ U.T @ b

    # the last element is always 1
    H = homogenous(H_vec).reshape(3, 3)
    H = np.linalg.inv(imgNormMat) @ H @ objNormMat
    H /= H[2, 2]

    H = _optimizeHomography(H, objPts, imgPts)

    return H


def _getNormalizationMatrix(pts: List[Vector2d], var: float = 2.0) -> Matrix3d:
    # calculate mean
    miu_x = 0.0
    miu_y = 0.0
    for p in pts:
        miu_x += p[0]
        miu_y += p[1]
    miu_x /= len(pts)
    miu_y /= len(pts)

    # calculate std
    std_x = 0.0
    std_y = 0.0
    for p in pts:
        std_x += (p[0] - miu_x) * (p[0] - miu_x)
        std_y += (p[1] - miu_y) * (p[1] - miu_y)
    std_x = math.sqrt(std_x / len(pts))
    std_y = math.sqrt(std_y / len(pts))

    # Compose normalization matrix
    sx = math.sqrt(var) / std_x
    sy = math.sqrt(var) / std_y
    normMat = np.array([
        [sx, 0, -sx * miu_x],
        [0, sy, -sy * miu_y],
        [0, 0, 1]
    ])

    return normMat


def _optimizeHomography(H: Matrix3d,
                        objPts: List[Vector3d],
                        imgPts: List[Vector2d],
                        verbose: bool = False) -> Matrix3d:
    """
    Optimize H using Levenberg-Marquardt algorithm.
    """
    # H has 8 DOF
    H_init = H.flatten()[:8]

    res: opt.OptimizeResult = opt.least_squares(
        _reprojection_error,
        H_init,
        _jacobian,
        method="lm",
        verbose=verbose,
        args=(objPts, imgPts),
    )
    H_final = homogenous(res.x).reshape(3, 3)

    return H_final


def _reprojection_error(H_vec: VectorXd,
                        objPts: List[Vector3d],
                        imgPts: List[Vector2d]) -> VectorXd:
    residuals: VectorXd = np.zeros([2 * len(objPts)], dtype=np.float64)

    H: Matrix3d = homogenous(H_vec).reshape(3, 3)
    for i in range(len(objPts)):
        objPt: Vector3d = homogenous(objPts[i][:2])
        predPt: Vector3d = H @ objPt
        predPt /= predPt[2]

        # add residuals:
        # Pi = [Xi, Yi, 1]
        # u = h1^T * Pi
        # v = h2^T * Pi
        # w = h3^T * Pi
        # xi' = u / w
        # yi' = v / w
        # Jx = xi - xi'
        # Jy = yi - yi'
        residuals[2 * i] = imgPts[i][0] - predPt[0]
        residuals[2 * i + 1] = imgPts[i][1] - predPt[1]

    return residuals


def _jacobian(H_vec: VectorXd,
              objPts: List[Vector3d],
              imgPts: List[Vector2d]) -> MatrixXd:
    jac: MatrixXd = np.zeros([2 * len(objPts), len(H_vec)], dtype=np.float64)

    H = homogenous(H_vec).reshape(3, 3)
    for i in range(len(objPts)):
        objPt: Vector3d = homogenous(objPts[i][:2])
        predPt: Vector3d = H @ objPt
        u: float = predPt[0]
        v: float = predPt[1]
        w: float = predPt[2]

        # ∂Jx/∂h
        jac[2 * i, 0:3, None] = -objPt / w
        jac[2 * i, 3:6, None] = 0.0
        jac[2 * i, 6:8, None] = u / (w * w) * objPt[:2]

        # ∂Jy/∂h
        jac[2 * i + 1, 0:3, None] = 0.0
        jac[2 * i + 1, 3:6, None] = -objPt / w
        jac[2 * i + 1, 6:8, None] = v / (w * w) * objPt[:2]

    return jac
