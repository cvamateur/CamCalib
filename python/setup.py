from setuptools import setup, find_packages


setup(
    name="camera_calibration",
    version="0.1.0",
    packages=find_packages(include=["camcalib", "camcalib/*"]),
)

