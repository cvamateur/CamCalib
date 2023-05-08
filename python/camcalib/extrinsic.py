import numpy as np
import cv2 as cv

from typing import List, Tuple
from .types import *

__all__ = ["workExtrinsics"]


def workExtrinsics(H_lst: List[Matrix3d],
                   K: Matrix3d) -> Tuple[List[Vector3d], List[Vector3d]]:
    n_imgs = len(H_lst)
    rvecs = [None] * n_imgs
    tvecs = [None] * n_imgs

    K_inv: Matrix3d = np.linalg.inv(K)
    R: Matrix3d = np.eye(3, dtype=np.float64)

    for i in range(n_imgs):
        h1: Vector3d = H_lst[i][:, 0:1]
        h2: Vector3d = H_lst[i][:, 1:2]
        h3: Vector3d = H_lst[i][:, 2:3]

        factor = 1. / np.linalg.norm(K_inv @ h1)

        r1: Vector3d = factor * K_inv @ h1
        r2: Vector3d = factor * K_inv @ h2
        r3: Vector3d = np.cross(r1.T, r2.T).T

        R[:, 0:1] = r1
        R[:, 1:2] = r2
        R[:, 2:3] = r3

        rvec, _ = cv.Rodrigues(R)
        tvec = factor * K_inv @ h3

        rvecs[i] = rvec
        tvecs[i] = tvec

    return rvecs, tvecs
