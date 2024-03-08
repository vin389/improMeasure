import numpy as np
def removeNanPoints(objPoints, imgPoints):
    # append nan points in a list
    nanPoints = []
    for i in range(imgPoints.shape[0]):
        for j in range(imgPoints.shape[1]):
            if np.isnan(imgPoints[i,j]):
                nanPoints.append(i)
                break
    for i in range(objPoints.shape[0]):
        for j in range(objPoints.shape[1]):
            if np.isnan(objPoints[i,j]):
                nanPoints.append(i)
                break
    nanPoints = list(set(nanPoints))        
    # remove rows of nan points from imgPoints and objPoints
    newImgPoints = imgPoints.copy()
    newObjPoints = objPoints.copy()
    newImgPoints = np.delete(newImgPoints, nanPoints, 0)        
    newObjPoints = np.delete(newObjPoints, nanPoints, 0)
    return newObjPoints, newImgPoints