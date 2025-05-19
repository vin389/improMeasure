
def syncByTriangulationMultiPoints(
        imgPoints, # imgPoints[icam][ipoint][istep] = [xi, yi]
                   # imgPoints[icam][ipoint] is a 2D numpy array sized (nt_icam,2)
                   # where nt_icam is the number of steps for the ith camera
                   # The number of steps varies by camera
        cmats,     # cmats[icam] is the 3-by-3 camera matrix of the ith camera
        dvecs,     # dvecs[icam] is the np-by-1 distortion vector of the ith camera
        rvecs,     # rvecs[icam] is the 3-by-1 rotation vector of the ith camera
        tvecs,     # tvecs[icam] is the 3-by-1 translational vector of the ith camera
        lagBounds, # lagBounds[icam][0:2] is the lower and upper bounds of the lag for the ith camera
                   # lagBounds[icam][0] is the lower bound of the lag for the ith camera
                   # lagBounds[icam][1] is the upper bound of the lag for the ith camera
                   # The lag is how many frames (floating point numbers) the ith camera started later 
                   # than the first camera (i.e., icam is 0).  
                   # For example, if lagBounds[1][0:2] = [1.0, 20.0], it means that we believe the 
                   # the 2nd (0-based) camera started recording n frames after the 1st camera started recording.
                   # And n could be between 1 and 20 frames.
                   # That is, for example if n is 1, it means the 100-th frame of camera 1 was taken 
                   # at the same time as the 99-th frame of camera 2 (as camera 2 started later so it 
                   # missed the 1st frame).
                   # lagBounds[0][0:2] must be [0.0, 0.0] because the first camera is the reference camera.
    ):
    pass


    



