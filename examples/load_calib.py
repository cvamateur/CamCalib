#!/usr/bin/env python
import os
import sys
import numpy as np
import cv2 as cv

from camcalib import load


def main(args):
    if not len(args) == 2:
        sys.stderr.write("usage: %s </path/to/calib.pkl>\n" % sys.argv[0])
        sys.exit(-1)
    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("error: path not exists: %s\n" % sys.argv[1])
        sys.exit(-1)

    with open(sys.argv[1], "rb") as f:
        res = load(f)

    assert isinstance(res, dict)
    for k, v in res.items():
        print(k, ":")
        print(v)
        print()


if __name__ == '__main__':
    main(sys.argv)
