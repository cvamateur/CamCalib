import numpy as np

from typing import List
from .types import *

__all__ = ["workIntrinsic"]


def _v(H, i, j):
    v = np.zeros([6], dtype=np.float64)
    v[0] = H[0, i] * H[0, j]
    v[1] = H[1, i] * H[0, j] + H[0, i] * H[1, j]
    v[2] = H[2, i] * H[0, j] + H[0, i] * H[2, j]
    v[3] = H[1, i] * H[1, j]
    v[4] = H[2, i] * H[1, j] + H[1, i] * H[2, j]
    v[5] = H[2, i] * H[2, j]
    return v


def workIntrinsic(H_lst: List[Matrix3d]) -> Matrix3d:
    assert len(H_lst) >= 3
    n_imgs = len(H_lst)

    # Compose V
    V: MatrixXd = np.zeros([2 * n_imgs, 6], dtype=np.float64)
    for i in range(n_imgs):
        V[2 * i, :] = _v(H_lst[i], 0, 1)
        V[2 * i + 1, :] = _v(H_lst[i], 0, 0) - _v(H_lst[i], 1, 1)

    # Solve B
    B: Matrix3d
    b: VectorXd
    if n_imgs == 3:
        raise NotImplementedError
    else:
        U: MatrixXd
        D: MatrixXd
        Vt: MatrixXd

        U, D, Vt = np.linalg.svd(V)
        b = Vt[-1, :]

    B = np.array([
        b[0], b[1], b[2],
        b[1], b[3], b[4],
        b[2], b[4], b[5],
    ]).reshape(3, 3)

    # Decompose B to K
    L: Matrix3d
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(-B)
    K = np.linalg.inv(L.T)
    K /= K[2, 2]
    return K
