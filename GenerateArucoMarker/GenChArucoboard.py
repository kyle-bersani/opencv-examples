import cv2
import cv2.aruco as aruco

squaresX=5
squaresY=7
# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 5 wide X 7 tall
gridboard = aruco.CharucoBoard([squaresX,squaresY], 
        squareLength=0.04, 
        markerLength=0.02, 
        dictionary=aruco.getPredefinedDictionary(aruco.DICT_5X5_1000))

# Create an image from the gridboard
img = gridboard.generateImage(outSize=(988, 1400))
cv2.imwrite("test_charuco.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()
