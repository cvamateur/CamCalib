import os
import sys
import pickle
import numpy as np

from typing import NamedTuple, List, Any

from .types import *


class CalibrationResult(NamedTuple):
    cameraMatrix: Matrix3d  # Camera intrinsic matrix.
    distCoeffs: VectorXd  # Distortion parameters
    rvecs: List[Vector3d]  # Rotation vectors of each image.
    tvecs: List[Vector3d]  # Translation vector of each image.


def to_dict(res: CalibrationResult):
    return dict(zip(res._fields, res))


def save(obj, f_):
    p = pickle.Pickler(f_)
    p.dump(obj)


def load(f_):
    p = pickle.Unpickler(f_)
    return p.load()
