import time
import cv2 as cv
import numpy as np 

fname='binFileTest.bin'

f = open(fname, 'rb')
cv.imshow('DUMMY', np.zeros((100,1000), dtype=np.uint8))
while True:
    data = f.read(256)
    f.seek(0)
    print('Data length: %3d: ' % len(data), end='')
    nprint = min(10, len(data))
    for i in range(nprint):
        print('%02x ' % data[i], end='')
    print()
    ikey = cv.waitKey(1000)
    if ikey == 32 or ikey == 27:
        break
    
    
    
cv.destroyWindow('DUMMY')
f.close()

 
       