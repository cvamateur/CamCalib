#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import cv2 as cv

WINNAME = "Collect Images"
PATH_FMT = ""
INDEX = 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="0", help="Camera id or video path.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("-n", "--prefix", default="img", help="Prefix of the saved image files.")
    return parser


def prepare_args(args):
    ind = args.input
    if len(ind) == 1 and ind not in ['.', '~', '/', ]:
        try:
            args.input = int(ind)
        except ValueError:
            sys.stderr.write(f"error: unknown camera id: {ind}\n")
            sys.exit(0)
    else:
        if not os.path.exists(ind):
            sys.stderr.write(f"error: video path not exists: {ind}\n")
            sys.exit(0)

    out_dir = os.path.abspath(os.path.expanduser(args.output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.output = out_dir
    sys.stdout.write(f"info: output directory: {out_dir}\n")

    global PATH_FMT
    PATH_FMT = os.path.join(out_dir, f"{args.prefix}_%04d.jpg")


def keyboard_signal(kid: int, frame) -> bool:
    if kid == ord(' '):
        global INDEX
        path = PATH_FMT % INDEX
        INDEX += 1
        cv.imwrite(path, frame)
        sys.stdout.write(f"info: save image: {os.path.basename(path)}\n")
        return True

    return kid != ord('q')   # press q to quit


def main(args):
    prepare_args(args)

    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        sys.stderr.write(f"error: cannot open stream: {args.input}\n")
        sys.exit(0)

    cv.namedWindow(WINNAME)
    try:
        while True:
            status, frame = cap.read()
            if not status:
                sys.stdout.write("info: stream closed\n")
                break

            cv.imshow(WINNAME, frame)
            if not keyboard_signal(cv.waitKey(10) & 0xFF, frame):
                break
    finally:
        cap.release()
    sys.stdout.write("info: finish\n")


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
