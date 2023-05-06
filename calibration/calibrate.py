from typing import List, Tuple
import numpy as np
import cv2 as cv
from numpy.typing import NDArray

_dtype = np.float64


def _v(H, i, j):
    v = np.zeros([6], dtype=_dtype)
    v[0] = H[0, i] * H[0, j]
    v[1] = H[1, i] * H[0, j] + H[0, i] * H[1, j]
    v[2] = H[2, i] * H[0, j] + H[0, i] * H[2, j]
    v[3] = H[1, i] * H[1, j]
    v[4] = H[2, i] * H[1, j] + H[1, i] * H[2, j]
    v[5] = H[2, i] * H[2, j]
    return v


def calibrate(objPts: List[NDArray],
              imgPts: List[NDArray],
              imgSize: Tuple[int, int]):
    """
    Calibrate camera using Zhang's method.

    @Params:
        objPts: Object points in 3D world.
        imgPts: Image points in pixels.
        imgSize: (Width, Height) of image.

    @Return
        camMat: Camera intrinsic matrix.
        dist: Lens distortion (Currently is None).
        rvecs: Rotation vectors of each image.
        tvecs: Translation vector of each image.
    """
    assert len(objPts) == len(imgPts) and len(objPts) >= 3

    #############
    # Intrinsic #
    #############
    V = np.zeros([2 * len(objPts), 6], dtype=_dtype)
    H = np.zeros([len(objPts), 3, 3], dtype=_dtype)
    for i, (objPts_i, imgPts_i) in enumerate(zip(objPts, imgPts)):
        # for each image, solve homography H that maps Pi=(Xi,Yi,1) to
        # p=(xi,yi,1). This can be solved by constructing linear equations
        # for all pairs of points in the image, each pair generate two equations:
        #   (Pi^T, 0^T, -xi*Pi^T) * H = 0
        #   (0^T, Pi^T, -yi*Pi^T) * H = 0
        # thus the linear system contains 2*num_points equations.
        # Since H has 8 DOFs, we need at least 4 points per image to solve H.
        assert len(objPts_i) >= 4 and len(objPts_i) == len(imgPts_i)
        objPts_i = np.asarray(objPts_i).reshape([-1, 3])
        imgPts_i = np.asarray(imgPts_i).reshape([-1, 2])

        A = np.zeros([2 * len(objPts_i), 8], dtype=_dtype)
        b = np.full([2 * len(objPts_i), 1], -1, dtype=_dtype)
        for j, (objPt, imgPt) in enumerate(zip(objPts_i, imgPts_i)):
            Pi = np.array([objPt[0], objPt[1], 1])  # [Xi, Yi, 1]
            A[2 * j, 0:3] = Pi
            A[2 * j, 3:6] = np.zeros([3], dtype=_dtype)
            A[2 * j, 6:8] = -imgPt[0] * Pi[0:2]
            b[2 * j] = -imgPt[0]

            A[2 * j + 1, 0:3] = np.zeros([3], dtype=_dtype)
            A[2 * j + 1, 3:6] = Pi
            A[2 * j + 1, 6:8] = -imgPt[1] * Pi[0:2]
            b[2 * j + 1] = -imgPt[1]

        # solve Ax=b
        if len(objPts_i) == 4:
            # TODO: Check the rank of A
            Hi = np.linalg.inv(A) * b
        else:
            u, s, vt = np.linalg.svd(A)
            s_inv = np.where(s < 1e-6, 0.0, 1.0 / s)
            s_inv_diag = np.zeros([8, 2 * len(objPts_i)], dtype=_dtype)
            s_inv_diag[range(8), range(8)] = s_inv
            Hi = vt.T @ s_inv_diag @ u.T @ b

        Hi = Hi.reshape(-1)
        Hi = np.append(Hi, 1).reshape(3, 3)

        # determine the sign
        # TODO: better way?
        sign = 1.0
        for k in range(len(objPts_i)):
            P_k = objPts_i[k].reshape(-1, 1)
            pred_px = Hi[k, :] @ np.array([P_k[0], P_k[1], 1], dtype=_dtype)
            if pred_px != 0.0:
                sign = -1.0 if pred_px < 0 else -1.0
                break

        H[i] = sign * Hi

        # We need decompose H to K, R, T. We can write:
        #   H = [h1, h2, h3] = K * [r1, r2, t], where
        #       hi: the i-th column of H, hi=[H_1i, H_2i, H_3i]^T
        #       r1: the first column of R
        #       r2: the second column of H
        #       t: translation vector
        # Expand the formula above, and multiply K_inv of the left:
        #   K_inv * h1 = r1
        #   K_inv * h2 = r2
        #   K_inv * h3 = t
        # Since r1, r2 stem from a rotation matrix, we exploit properties:
        #   r1^T * r2 = 0
        #   |r1| = |r2| = 1 -> r1^T * r1 - r2^T * r2 = 0
        # Substitute in to the properties we get:
        #   h1^T * K_inv^T * K_inv * h2 = 0
        #   h1^T * K_inv^T * K_inv * h1 - h2^T * K_inv^T * K_inv * h2 = 0
        # To simplify the equations, we define a symmetric and positive
        # definite matrix B = K_inv^T * K_inv, such that:
        #   h1^T * B * h2 = 0
        #   h1^T * B * h1 - h2^T * B * h2 = 0
        # If B is solved, then K is obtained by Cholesky decomposition:
        #   Chol(B) = L * L^T, where L = K_inv^T
        # Write the equations relative to B into a linear system as:
        #  V_12^T * b = 0
        #  V_11^T * b - V_22^T * b = 0, where
        #       b = [b11, b12, b13, b12, b22, b23, b13, b23, b33]^T
        #       V_ij = [ H_1i*H_1j,
        #                H_2i*H_1j + H_1i * H_2j,
        #                H_3i*H_1j + H_1i * H_3j,
        #                H_2i * H_2j,
        #                H_3i * H_2j + H_2i * H_3j,
        #                H_3i * H_3j]^T
        # Since B has 5/6 DOF, we need at least 3 different views.
        for i, Hi in enumerate(H):
            V[2 * i, :] = _v(Hi, 0, 1)
            V[2 * i + 1, :] = _v(Hi, 0, 0) - _v(Hi, 1, 1)

    # Solve B
    if len(objPts) == 3:
        # TODO
        raise NotImplementedError
    else:
        u, s, vt = np.linalg.svd(V)
        b = vt[-1, :]
    B = np.array([
        b[0], b[1], b[2],
        b[1], b[3], b[4],
        b[2], b[4], b[5]
    ]).reshape(3, 3)

    # Decompose B to K
    L = np.linalg.cholesky(B)
    camMat = np.linalg.inv(L.T)
    camMat /= camMat[2, 2]

    #############
    # Extrinsic #
    #############
    rvecs, tvecs = [], []

    K_inv = np.linalg.inv(camMat)
    for i, Hi in enumerate(H):
        h1 = Hi[:, 0].reshape(3, 1)
        h2 = Hi[:, 1].reshape(3, 1)
        h3 = Hi[:, 2].reshape(3, 1)

        r1 = K_inv @ h1
        rho = 1.0 / np.linalg.norm(r1)
        r1 *= rho
        r2 = rho * K_inv @ h2
        r3 = np.cross(r1.T, r2.T).T
        R = np.concatenate([r1, r2, r3], axis=1)

        rvec, _ = cv.Rodrigues(R)
        tvec = rho * K_inv @ h3

        rvecs.append(rvec)
        tvecs.append(tvec)

    return camMat, None, rvecs, tvecs
