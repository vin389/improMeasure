import cv2
from pickTemplates import pickTemplates

my_img = cv2.imread('d:/temp/test.jpg')

x = pickTemplates(img=my_img, nPoints=1)




