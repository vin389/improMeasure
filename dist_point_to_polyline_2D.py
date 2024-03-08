import numpy as np
from numba import njit 

@njit
def dist_point_to_polyline_2D(p, polyline):
    '''
    The function calculates the minimum distance between point p1 and 
    the polyline. 
    The result is returned as a float representing the minimum distance, 
    the closest point on the polyline, the segment that the closest point is
    on, and the coefficient alpha (which is between 0 and 1)
    This function assumes the polyline is open. If the user wants it to be 
    a closed polygon, user can concatenate the first vertex to the polyline.

    Parameters
    ----------
    p : list or a numpy array, just only two elements
        the point.
    polyline : numpy.array (N by 2)
        the polyline.

    Returns
    -------
    [0]: the minimum distance
    [1]: the closest point on the polyline that is closest to the point
    [2]: the index of the segment which the closest point is on. 
         0 means the segment is between vertices 0 and 1.
         the index must between 0 and N - 2
    [3]: the alpha. 
         0.0 means the closest point is at the vertex of index returned[2]     
    
    Example 1:
    --------
    point = np.array([2, 3.])
    polyline = np.array([-1,-1,-1,1,1,1,1,-1,   -1,-1.]).reshape(-1, 2)
    dist, point, seg, alpha = dist_point_to_polyline_2D(point, polyline)
    # where
    # dist is 2.236 
    # point is array([1., 1.])
    # seg is 1
    # alpha is 1.0    

    Example 2:
    --------
    point = np.array([0.8, 0.8])
    polyline = np.array([-1,-1,-1,1,1,1,1,-1,  -1, -1.]).reshape(-1, 2)
    dist, point, seg, alpha = dist_point_to_polyline_2D(point, polyline)
    # where
    # dist is 0.2
    # point is array([.8, 1.])
    # seg is 1
    # alpha is .9    
    '''
    # This statement cannot be optimized by @njit
    # probably because of too many candidate implementation
#    polygon = np.array(polygon, dtype=float).reshape(-1, 2)
    n = polyline.shape[0]
    minDist, closestPt, bestAlpha = dist_point_to_line_2D(
        p, polyline[0, :].flatten(), polyline[1, :].flatten())
    segment = 0
    for i in range(1, n - 1):
#        dist, pc, alpha = dist_point_to_line_2D(
#            p, polyline[i, :].flatten(), polyline[i + 1, :].flatten())
        # calculate minimum distance (minDist), closest point (closestPt), 
        # and the alpha of the closest point
        pa = polyline[i, :].flatten()
        pb = polyline[i + 1, :].flatten()
        denominator = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
        # if pa and pb are about the same points
        if denominator > 1e-16:
            # 
            numerator = (pa[0] - p[0]) * (pa[0] - pb[0]) \
                      + (pa[1] - p[1]) * (pa[1] - pb[1])
            # alpha
            alpha = numerator / denominator
            alpha = min(1., max(0., alpha))
            # pc
            pc = pa.copy()
            pc[0] = (1. - alpha) * pa[0] + alpha * pb[0]
            pc[1] = (1. - alpha) * pa[1] + alpha * pb[1]
        else:
            pc = pa
            alpha = 0.0
        dist = ((pc[0] - p[0]) ** 2 + (pc[1] - p[1]) ** 2) ** .5
        # update the best one so far
        if dist < minDist:
            minDist = dist
            closestPt = pc.copy()
            bestAlpha = alpha
            segment = i
    return minDist, closestPt, segment, bestAlpha


# def test_dist_point_to_polygon():
#     p1 = np.array([0.,1], dtype=float)
#     polygon = polygon = np.array([-3., 0., -2, 0., 1., 0.]).reshape(3, 2)
#     print("The min. dist. is ", dist_point_to_polygon_2D(p1, polygon))
    
# test_distance_point_to_polygon()

@njit
def dist_point_to_line_2D(p, pa, pb):
    # p, pa, and pb must be 2D points that can be accessed by [0] and [1]
    # such as a 1D numpy.array or a list. (but cannot be tuple)
    # for example p[0] and p[1] 
    # This function returns a tuple:
    # [0]: the distance
    # [1]: the point (pc) that is between pa and pb and has the min. distance
    # [2]: the coefficient alpha which pc = (1 - alpha)pa + alpha pb
    #      the coefficient must be between 0 and 1.
    # Example:
    #   dist_point_to_line_2D([0, 1.], [-1,0], [3,0])
    #       returns
    #       (1., [0., 0.], 0.25)
    denominator = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
    # if pa and pb are about the same points
    if denominator > 1e-16:
        # 
        numerator = (pa[0] - p[0]) * (pa[0] - pb[0]) \
                  + (pa[1] - p[1]) * (pa[1] - pb[1])
        # alpha
        alpha = numerator / denominator
        alpha = min(1., max(0., alpha))
        # pc
        pc = pa.copy()
        pc[0] = (1. - alpha) * pa[0] + alpha * pb[0]
        pc[1] = (1. - alpha) * pa[1] + alpha * pb[1]
    else:
        pc = pa
        alpha = 0.0
    dist = ((pc[0] - p[0]) ** 2 + (pc[1] - p[1]) ** 2) ** .5
    return dist, pc, alpha
    

def dist_point_to_line_3D(p, pa, pb):
    # p, pa, and pb must be 3D points that can be accessed by [0], [1], and [2]
    # such as a 1D numpy.array or a list. (but cannot be tuple)
    # for example p[0] and p[1] 
    # This function returns a tuple:
    # [0]: the distance
    # [1]: the point (pc) that is between pa and pb and has the min. distance
    # [2]: the coefficient alpha which pc = (1 - alpha)pa + alpha pb
    #      the coefficient must be between 0 and 1.
    # Example:
    #   dist_point_to_line_3D([0, 1., 0], [-1,0, 0], [3,0,0])
    #       returns
    #       (1., [0., 0., 0.], 0.25)
    denominator = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
    # if pa and pb are about the same points
    if denominator > 1e-16:
        # 
        numerator = (pa[0] - p[0]) * (pa[0] - pb[0]) \
                  + (pa[1] - p[1]) * (pa[1] - pb[1]) \
                  + (pa[2] - p[2]) * (pa[2] - pb[2])
        # alpha
        alpha = numerator / denominator
        alpha = min(1., max(0., alpha))
        # pc
        pc = pa.copy()
        pc[0] = (1. - alpha) * pa[0] + alpha * pb[0]
        pc[1] = (1. - alpha) * pa[1] + alpha * pb[1]
        pc[2] = (1. - alpha) * pa[2] + alpha * pb[2]
    else:
        pc = pa
        alpha = 0.0
    dist = ((pc[0] - p[0]) ** 2 + (pc[1] - p[1]) ** 2 + (pc[2] - p[2]) ** 2) ** .5
    return dist, pc, alpha
    
@njit 
def minDistPointToPolygonalchain2d(thePoint, thePoly):
    """ Finds the minimum distance between a point (thePoint) and a polygonal
    chain. 
    
    Parameters:
        thePoint: the 2D coordinate of the point. It can be a Numpy array, a 
            list, or a tuple. The thePoint[0] is x and [1] is y.
        thePoly: the 2D coordinates of the vertices that compose the polygonal
            chain. It is an N-by-2 polygonal chain. 
        
    Returns: 
        [0]: the minimum distance
        [1]: np.array of the closest point 
            
    """
    # 
#    thePoint = np.array(thePoint).reshape(-1)
#    thePoly = np.array(thePoly)
#    thePoly = thePoly.reshape(int(thePoly.size / 2), 2)
    nPoints = thePoly.shape[0]
    
    # this section of code calculate the minDistI 
    i = 0;
#    minDist = 1e30
    minDistSq = 1e30
    x5,y5 = 0., 0.
    minSeg = -1
    minAlpha = 0.
    for i in range(nPoints - 1):   
#        minDistI = 1e30
        minDistISq = 1e30
        p1 = thePoly[i, :]
        p2 = thePoly[i + 1, :]
        p3 = thePoint
        x1,y1 = p1[0], p1[1]
        x2,y2 = p2[0], p2[1]
        x3,y3 = p3[0], p3[1]
        
        alpha_denomi = (x2 - x1) ** 2 + (y2 - y1) ** 2
        alpha_numera = (x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1) 
        if abs(alpha_denomi) < 1e-12: 
            # If p1 and p2 stick together, the distance is between p1 and p3
            alpha = 0.0
            x4, y4 = x1, y1
        else:
            alpha = alpha_numera / alpha_denomi
            if alpha < 0.0:
                # distance is between p1 and p3
                x4, y4 = x1, y1
            elif alpha > 1.0:
                # distance is between p2 and p3
                x4, y4 = x2, y2
            else:    
                # distance is between p4 and p3, where p4 is between p1 and p2
                x4 = (1. - alpha) * x1 + alpha * x2
                y4 = (1. - alpha) * y1 + alpha * y2
        # if minDistI is smaller than minDist
#        minDistI = ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** .5
        minDistISq = (x4 - x3) ** 2 + (y4 - y3) ** 2
#        if (minDistI < minDist):
#            minDist = minDistI
        if (minDistISq < minDistSq):
            minDistSq = minDistISq
            x5, y5 = x4, y4
            minSeg = i
            minAlpha = alpha            
#    return minDist, np.array([x5, y5]) 
    return np.sqrt(minDistSq), np.array([x5, y5]), minSeg, minAlpha 
