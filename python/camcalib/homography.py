import math
import numpy as np

from typing import List
from .types import *

__all__ = ["findHomography"]


def getNormalizationMatrix(pts: List[Vector2d], var: float = 2.0) -> Matrix3d:
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


def homogenous(x: VectorXd) -> VectorXd:
    return np.append(x, 1).astype(np.float64)


def hnormalize(x: VectorXd) -> VectorXd:
    x /= x[-1]
    return x[:-1].astype(np.float64)


def findHomography(objPts: List[Vector3d], imgPts: List[Vector2d]) -> Matrix3d:
    assert len(objPts) == len(imgPts) and len(objPts) >= 4
    objNormMat = getNormalizationMatrix(objPts)
    imgNormMat = getNormalizationMatrix(imgPts)

    n_pts = len(objPts)
    A: MatrixXd = np.zeros([2 * n_pts, 8])
    b: VectorXd = np.zeros([2 * n_pts, 1])
    for i in range(n_pts):
        objPt: Vector3d = homogenous(objPts[i][:2]).reshape(-1, 1)
        imgPt: Vector3d = homogenous(imgPts[i]).reshape(-1, 1)

        objPt = objNormMat @ objPt
        imgPt = imgNormMat @ imgPt

        A[2 * i, 0:3] = objPt.T
        A[2 * i, 3:6] = np.zeros([3])
        A[2 * i, 6:8] = -imgPt[0] * objPt[0:2].T
        b[2 * i, 0] = imgPt[0]

        A[2 * i + 1, 0:3] = np.zeros([3])
        A[2 * i + 1, 3:6] = objPt.T
        A[2 * i + 1, 6:8] = -imgPt[1] * objPt[0:2].T
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

    H = homogenous(H_vec).reshape(3, 3)
    H = np.linalg.inv(imgNormMat) @ H @ objNormMat
    H /= np.linalg.norm(H)
    return H
