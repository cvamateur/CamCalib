import math
import numpy as np
import cv2 as cv

from typing import List
from .types import *

__all__ = ["workDistortion"]


def workDistortion(camMat: Matrix3d,
                   rvecs: List[Vector3d],
                   tvecs: List[Vector3d],
                   objPts: List[List[Vector3d]],
                   imgPts: List[List[Vector2d]]) -> VectorXd:

    fx = camMat[0, 0]
    fy = camMat[1, 1]
    cx = camMat[0, 2]
    cy = camMat[1, 2]

    n_imgs = len(objPts)
    n_pts = len(objPts[0])

    # Compose A
    A: MatrixXd = np.zeros([2 * n_imgs * n_pts, 5], dtype=np.float64)
    b: VectorXd = np.zeros([2 * n_imgs * n_pts, 1], dtype=np.float64)
    for i in range(n_imgs):
        R, _ = cv.Rodrigues(rvecs[i])
        t = tvecs[i]

        for j in range(n_pts):
            Pw = objPts[i][j].reshape(3, 1)

            # points undistorted
            Pc = (R @ Pw + t).flatten()
            u = Pc[0] / Pc[2]
            v = Pc[1] / Pc[2]
            r = math.sqrt(u * u + v * v)

            # points distorted
            up = (imgPts[i][j][0] - cx) / fx
            vp = (imgPts[i][j][1] - cy) / fy

            A[2 * i + j] = np.array([u * pow(r, 2), u * pow(r, 4), u * pow(r, 6), 2 * u * v, r * r + 2 * u * u])
            b[2 * i + j] = up - u

            A[2 * i + j + 1] = np.array([v * pow(r, 2), v * pow(r, 4), v * pow(r, 6), r * r + 2 * v * v, 2 * u * v])
            b[2 * i + j + 1] = vp - v

    u, s, vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.zeros([5, 5], dtype=np.float64)
    for i in range(5):
        s_inv[i, i] = 0. if s[i] < 1e-6 else 1. / s[i]

    distCoeffs = vt.T @ s_inv @ u.T @ b
    distCoeffs = distCoeffs.flatten()
    return distCoeffs
