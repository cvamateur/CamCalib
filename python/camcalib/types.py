import sys
import numpy as np
from typing import Generic, TypeVar, Tuple, Literal

__all__ = [
    "DType", "Size", "Matrix",
    "MatrixXd", "Matrix2d", "Matrix3d", "Matrix4d",
    "VectorXd", "Vector2d", "Vector3d", "Vector4d",
]

DType = TypeVar("DType", bound=np.generic)
Size = Tuple[int, int]

if sys.version_info >= (3, 9):
    from typing import Annotated
    from numpy.typing import NDArray

    Matrix = Annotated[NDArray[DType], Literal["int", "int"]]

    MatrixXd = Annotated[NDArray[np.float64], Literal[-1, -1]]
    Matrix2d = Annotated[NDArray[np.float64], Literal[2, 2]]
    Matrix3d = Annotated[NDArray[np.float64], Literal[3, 3]]
    Matrix4d = Annotated[NDArray[np.float64], Literal[4, 4]]

    VectorXd = Annotated[NDArray[np.float64], Literal[-1, 1]]
    Vector2d = Annotated[NDArray[np.float64], Literal[2, 1]]
    Vector3d = Annotated[NDArray[np.float64], Literal[3, 1]]
    Vector4d = Annotated[NDArray[np.float64], Literal[4, 1]]
else:
    Shape = TypeVar("Shape")

    class _NDArray(Generic[Shape, DType], np.ndarray):
        def __getitem__(self, item): ...


    Matrix = _NDArray["int,int", np.float64]
    MatrixXd = _NDArray["-1,-1", np.float64]
    Matrix2d = _NDArray["2,2", np.float64]
    Matrix3d = _NDArray["3,3", np.float64]
    Matrix4d = _NDArray["4,4", np.float64]

    VectorXd = _NDArray["-1,1", np.float64]
    Vector2d = _NDArray["2,1", np.float64]
    Vector3d = _NDArray["3,1", np.float64]
    Vector4d = _NDArray["4,1", np.float64]
