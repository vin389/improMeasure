import cv2 as cv
import numpy as np
import threading

def projectPoints_mp(objPoints, 
                      rvec, tvec, 
                      cmat, dvec):   
    objPoints = objPoints.reshape(-1,3)
    npts = objPoints.shape[0]
    nthread = 16
    imgPoints = np.zeros((npts, 1, 2), dtype=np.float64)
#    jacobi = np.zeros((2*npts, 6+4+dvec.size), dtype=np.float64)
    pointStart = []
    for i in range(nthread + 1):
        pointStart.append( (npts * i) // nthread)
        
    # def job(p0,p1):
    #     imgPoints[p0:p1,:,:], jacobi[2*p0:2*p1,:] =\
    #         cv.projectPoints(objPoints[p0:p1,:], rvec, tvec, cmat, dvec)
    def job(p0,p1):
        nBatch = 1000
        for p0B in range(p0, p1, nBatch):
            p1B = min(p0B + nBatch, p1)
            imgPoints[p0B:p1B,:,:], dummy =\
                cv.projectPoints(objPoints[p0B:p1B,:], rvec, tvec, cmat, dvec)

    threads = []
    for i in range(nthread):
        p0 = pointStart[i]
        p1 = pointStart[i + 1]
        threads.append(threading.Thread(target = job, args = (p0,p1)))
        threads[i].start()
#        job(p0, p1)
        # imgPoints[p0:p1,:,:], jacobi[2*p0:2*p1,:] =\
        #     cv.projectPoints(objPoints[p0:p1,:], rvec, tvec, cmat, dvec)
    for i in range(nthread):
        threads[i].join()

#    return imgPoints, jacobi
    return imgPoints
