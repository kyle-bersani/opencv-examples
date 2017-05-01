## Overview

The code in this directory is a test camera calibration done on a series of images I captured of a printed out Charuco Board. You can generate a similar charuco board using the generation scripts in this repository. This camera calibration is required for using Aruco camera posture estimation.
 
It is based off of the OpenCV-Python tutorial here: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html 
 
The calibration file to be used with posturing is saved to calibration.pckl
 
## Systems

This code was successfully run on the following system:
- Linux Debian dist
  - Linux Mint 18.1 Cinnamon 64-bit
  - Python 2.7 with OpenCV 3.2.0
- MacBook Air
  - macOS Sierra 10.12.4
  - Python 2.7 with OpenCV 3.2.0
