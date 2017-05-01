# The following code is used to watch a video stream, detect a Charuco board, and use
# it to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle

# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)
CHARUCOBOARD_ROWCOUNT=7
CHARUCOBOARD_COLCOUNT=5

# Create grid board object we're using in our stream
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture('charucoboard-test-at-kyles-desk.mp4')

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = CHARUCO_BOARD,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Only try to find CharucoBoard if we found markers
        if ids is not None and len(ids) > 10:

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=CHARUCO_BOARD)
    
            # Require more than 20 squares
            if response is not None and response > 20:
                # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video 
                pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners, 
                        charucoIds=charuco_ids, 
                        board=CHARUCO_BOARD, 
                        cameraMatrix=cameraMatrix, 
                        distCoeffs=distCoeffs)
                if pose:
                    # Draw the camera posture calculated from the gridboard
                    QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
            
        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
