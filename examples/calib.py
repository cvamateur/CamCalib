#!/usr/bin/env python
import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2 as cv

from camcalib import calibrate
from camcalib.results import save, to_dict


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Directory of chessboard images.")
    parser.add_argument("-o", "--output", default=f"./output", help="Directory of results to save.")
    parser.add_argument("-s", "--show-corners", action="store_true", help="Whether to display detected corners.")
    parser.add_argument("-ext", default="png", help="Image extension (default png).")
    parser.add_argument("--chessboard", default="11x7", help="Chessboard size counted by corners (default 11x7).")

    return parser


def main(args):
    """
    Refer to https://learnopencv.com/camera-calibration-using-opencv/ for more details.
    """
    if not os.path.exists(args.output):
        sys.stderr.write(f"[ERROR]: path not exist: {args.output}")
        sys.exit(-1)
    args.output = os.path.abspath(os.path.expanduser(args.output))

    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    FLAG = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    # chessboard dimension
    CHESSBOARD = [int(x) for x in args.chessboard.split('x')]

    # vector to store vectors of 3D points for each checkerboard image
    objpoints = []

    # vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # define the world coordinates for 3D points
    objp = np.zeros([CHESSBOARD[0] * CHESSBOARD[1], 3], dtype=np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    # get image paths
    path_pattern = os.path.join(args.input, f"*.{args.ext}")
    img_paths = glob.glob(path_pattern)
    if not img_paths:
        sys.stderr.write(f"error: no image found: {path_pattern}\n")
        sys.exit(-1)
    print(f"info: found {len(img_paths)} images")

    # cv.namedWindow("Corner", cv.WINDOW_FULLSCREEN)
    # cv.resizeWindow("Corner", 1024, 768)
    h, w = (0, 0)
    for path in img_paths:
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)
        print(path)
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        h, w = img_bgr.shape[:2]

        img_bgr_copy = img_bgr.copy()
        img_bgr_copy = cv.resize(img_bgr_copy, (1024, 768))
        cv.imshow("Image", img_bgr_copy)
        cv.waitKey(10)

        # find checkerboard corners
        # if desired number of corners are found in the image then ret=true
        ret, corners = cv.findChessboardCorners(img_gray, CHESSBOARD, FLAG)

        if ret:
            # refine checkerboard corners
            corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), CRITERIA)

            objpoints.append(objp)
            imgpoints.append(corners2.reshape(-1, 2))

            # display corners
            if args.show_corners:
                img_bgr = cv.drawChessboardCorners(img_bgr, CHESSBOARD, corners2, ret)
                img_bgr = cv.resize(img_bgr, (1024, 768), interpolation=cv.INTER_LINEAR)
                cv.imshow("Corner", img_bgr)
                cv.waitKey(20)
        else:
            sys.stderr.write(f"skip: no chessboard pattern detected: {path}\n")

    # perform camera calibration by
    # passing the value of known 3D points (objpoints)
    # and corresponding pixel coordinates of the
    # detected corners (imgpoints)
    print("start calibrating")
    res = calibrate(objpoints, imgpoints, (w, h))
    K, D, rvecs, tvecs = res
    out_fn = os.path.join(args.output, "calib.pkl")
    with open(out_fn, "wb") as f:
        save(to_dict(res), f)
    print(f"info: save calibration to: {out_fn}")

    print("Intrinsic Matrix:\n", K)
    print("Lens Distortion:\n", D)
    print("Extrinsic Rotation:\n", rvecs)
    print("Extrinsic Translation:\n", tvecs)

    # Use the derived camera parameters to undistort the image
    img_bgr = cv.imread(img_paths[0], cv.IMREAD_COLOR)

    # refining the camera matrix using parameters obtained by calibration
    K_new, roi = cv.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    # method 1
    dst1 = cv.undistort(img_bgr, K, D, None, K_new)

    # method 2
    mapx, mapy = cv.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv.CV_32FC1)
    dst2 = cv.remap(img_bgr, mapx, mapy, cv.INTER_LINEAR)

    cv.imshow("origin", img_bgr)
    cv.imshow("method 1", dst1)
    cv.imshow("method 2", dst2)
    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == '__main__':
    args = get_parser().parse_args()
    sys.stdout.write(json.dumps(args.__dict__, indent=2))
    sys.stdout.write('\n------------------------------\n')
    main(args)
