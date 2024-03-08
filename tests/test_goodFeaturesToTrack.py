import numpy as np
import cv2 as cv
from imshow2 import imshow2

# Read the image and convert to greyscale
img = cv.imread(r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen01_Canon_L_20220310_102517.JPG')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the top 20 corners using the cv2.goodFeaturesToTrack()
corners = cv.goodFeaturesToTrack(gray,20000,0.01,10)
corners = np.int0(corners)

# Iterate over the corners and draw a circle at that location
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),5,(0,0,255),-1)
    
# Display the image
imshow2('a', img)
cv.waitKey(0)