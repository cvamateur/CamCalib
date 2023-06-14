import os
import sys
import glob
import argparse
import numpy as np
import cv2 as cv

from camcalib import load


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Directory of chessboard images.")
    parser.add_argument("-f", "--file", required=True, help="Path of calib.pkl")
    parser.add_argument("-ext", default="png", help="Image extension (default png).")
    return parser


def main(args):
    assert os.path.exists(args.input)
    assert os.path.exists(args.file)

    # get image paths
    path_pattern = os.path.join(args.input, f"*.{args.ext}")
    img_paths = glob.glob(path_pattern)
    if not img_paths:
        sys.stderr.write(f"error: no image found: {path_pattern}\n")
        sys.exit(-1)
    sys.stdout.write(f"info: {len(img_paths)} images found\n")

    # get calibration result
    with open(args.file, "rb") as f:
        res = load(f)
    K = res.get("K", None)
    D = res.get("D", None)
    if K is None or D is None:
        sys.stderr.write("error: unknown calibration result\n")
        sys.exit(-1)

    cv.namedWindow("Undistorted", cv.WINDOW_NORMAL)
    cv.resizeWindow("Undistorted", 1600, 900)

    K_new = None
    w, h = 0, 0
    for i, path in enumerate(img_paths):
        print(path)
        img_bgr = cv.imread(path, cv.IMREAD_COLOR)

        if i == 0:
            h, w = img_bgr.shape[:2]
            K_new, roi = cv.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            mapx, mapy = cv.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv.CV_32FC1)

        img_final = cv.remap(img_bgr, mapx, mapy, cv.INTER_LINEAR)
        cv.imshow("Undistorted", img_final)
        k = cv.waitKey(0)
        if k == ord('q'):
            break


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
