# Camera Calibration 

Implementation of Zhang's method using C++ and Python.
- **Intrinsic Matrix** (complete)
- **Extrinsic Matrices** (complete)
- Radial distortion (in progress)
- Tangential distortion (in progress)

## Prerequisite
- OpenCV 4.5.5
- Eigen 3.4
- Ceres 2.2
- CUDA Toolkit

## Build C++ Version as Shared Library
```shell
mkdir build && cd build
cmake ..
make -j4
```

## Install Python Version as package
```shell
cd python
pip install -e .
```

## Run Camera Calibration (C++)
```shell
cd build/example

# Self implementation
./calib ../data/imgs/leftcamera 11 7

# OpenCV implementation
./calib_cv ../data/imgs/leftcamera 11 7
```

## Run Camera Calibration (Python3)
```shell
cd example

# Self implementation
python3 calib.py -i ../data/imgs/leftcamera

# OpenCV implementation
python3 calib_cv.py -i ../data/imgs/leftcamera
```

