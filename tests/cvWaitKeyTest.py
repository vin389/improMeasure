import numpy as np
import cv2 as cv
import datetime

img = np.zeros((300,300), dtype=np.uint8)

def imshowEvent(event,x,y,flags,param):
    nowStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    print("(%s) event:%d Flags:%d x:%d y:%d" % (nowStr, event, flags, x, y))

winTitle = 'Test OpenCV event'
cv.namedWindow(winTitle)
cv.setMouseCallback(winTitle, imshowEvent)

cv.imshow(winTitle, img)
while True:
    ikey = cv.waitKeyEx(0)
    nowStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    print("(%s) ikeyEx: (%d)" % (nowStr, ikey))
    if ikey == 27 or ikey == 'q':
        print("Quit.")
        break

try:
    cv.destroyWindow(winTitle)
except:
    pass