import cv2
import cv2.aruco as aruco

# Create gridboard, which is a set of Aruco markers
# the following call gets a board of markers 5 wide X 7 tall
gridboard = aruco.GridBoard_create(
        markersX=5, 
        markersY=7, 
        markerLength=0.04, 
        markerSeparation=0.01, 
        dictionary=aruco.Dictionary_get(aruco.DICT_5X5_1000))

# Create an image from the gridboard
img = gridboard.draw(outSize=(988, 1400))
cv2.imwrite("test_gridboard.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()

