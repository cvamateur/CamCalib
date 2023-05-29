from typing import List, NamedTuple

import numpy as np

from .homography import findHomography
from .intrinsic import workIntrinsic
from .extrinsic import workExtrinsics
from .distortion import workDistortion
from .refine_all import refine_all
from .results import CalibrationResult
from .types import *


def calibrate(objPts: List[List[Vector3d]],
              imgPts: List[List[Vector2d]],
              imgSize: Size):
    """
    Calibrate camera using Zhang's method.

    @Params:
        objPts: Object points in 3D world.
        imgPts: Image points in pixels.
        imgSize: (Width, Height) of image.

    @Return
        res: Calibration Result.
    """
    assert len(objPts) == len(imgPts) and len(objPts) >= 3
    n_imgs = len(objPts)

    #############
    # Intrinsic #
    #############
    # for each image, solve homography H that maps Pi=(Xi,Yi,1)^T to
    # p=(xi,yi,1)^T. This can be solved by constructing linear equations
    # for all pairs of points in the image, each pair generate two equations:
    #   (Pi^T, 0^T, -xi*Pi^T) * H = 0
    #   (0^T, Pi^T, -yi*Pi^T) * H = 0
    # thus the linear system contains 2*num_points equations.
    # Since H has 8 DOFs, we need at least 4 points per image to solve H.
    H_lst: List[Matrix3d] = [None] * n_imgs
    for i in range(n_imgs):
        H_lst[i] = findHomography(objPts[i], imgPts[i])

    # We need decompose H to K, R, T. We can write:
    #     H = [h1, h2, h3] = K * [r1, r2, t], where
    #        hi: the i-th column of H, hi=[H_1i, H_2i, H_3i]^T
    #        r1: the first column of R
    #        r2: the second column of R
    #        t: translation vector
    # Expand the formula above, and multiply K_inv from the left:
    #     K_inv * h1 = r1
    #     K_inv * h2 = r2
    #     K_inv * h3 = t
    # Since r1, r2 stem from a rotation matrix, we exploit the properties:
    #     r1^T * r2 = 0
    #     |r1| = |r2| = 1 -> r1^T * r1 - r2^T * r2 = 0
    # Substitute in to the properties we get:
    #     h1^T * K_inv^T * K_inv * h2 = 0
    #     h1^T * K_inv^T * K_inv * h1 - h2^T * K_inv^T * K_inv * h2 = 0
    # To simplify the equations, we define a symmetric and positive
    # definite matrix B = K_inv^T * K_inv, we can rewrite the above formulas:
    #     h1^T * B * h2 = 0
    #     h1^T * B * h1 - h2^T * B * h2 = 0
    # If B is solved, then K can be obtained by Cholesky decomposition:
    #     Chol(B) = L * L^T, where L = K_inv^T
    # Write the equations relative to B into a linear system as:
    #    V_12^T * b = 0
    #    V_11^T * b - V_22^T * b = 0, where
    #       b = [b11, b12, b13, b12, b22, b23, b13, b23, b33]^T
    #       V_ij = [ H_1i*H_1j,
    #                H_2i*H_1j + H_1i * H_2j,
    #                H_3i*H_1j + H_1i * H_3j,
    #                H_2i * H_2j,
    #                H_3i * H_2j + H_2i * H_3j,
    #                H_3i * H_3j]^T
    # Since B has 5/6 DOF, we need at least 3 different views.
    camMat = workIntrinsic(H_lst)

    #############
    # Extrinsic #
    #############
    #   ρ * H = K * [r1 r2 t]
    #   ρ * K_inv * H = [r1 r2 t]
    #   ρ = 1 / |K_inv * H|
    #   r1 = ρ * K_inv * h1
    #   r2 = ρ * K_inv * h2
    #   r3 = r1 x r2
    #   r = ρ * K_inv * h3
    rvecs, tvecs = workExtrinsics(H_lst, camMat)

    ##############
    # Distortions
    ##############
    # Initialize distortions by solving linear system: Ax=b
    # up - u = ur^2 * k1 + ur^4 * k2 + ur^6 * k3 + 2uv * p1 + (r^2 + 2u^2) * p2
    # vp - v = vr^2 * k1 + vr^4 * k2 + vr^6 * k3 + (r^2 + 2v^2) * p1 + 2uv * p2
    # where,
    #       up, vp, u, v, r are known
    #       k1, k2, k3, p1, p3 are unknowns
    # up and vp can be calculated by:
    #       up = (xi - cx) / fx
    #       vp = (yi - cy) / fy
    distCoeffs: VectorXd = workDistortion(camMat, rvecs, tvecs, objPts, imgPts)
    # distCoeffs: VectorXd = np.zeros([5], dtype=np.float64)

    #########
    # Refine
    #########
    refine_all(camMat, distCoeffs, rvecs, tvecs, objPts, imgPts)

    return CalibrationResult(camMat, distCoeffs, rvecs, tvecs)
