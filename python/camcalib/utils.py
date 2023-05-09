import numpy as np

from .types import *


def homogenous(x: VectorXd) -> VectorXd:
    x = np.asarray(x, dtype=np.float64)
    x = np.append(x, 1).reshape(-1, 1)
    return x


def hnormalize(x: VectorXd) -> VectorXd:
    x = np.asarray(x, dtype=np.float64)
    x /= x[-1]
    x = x[:-1]
    return x
