import numpy as np 
import matplotlib.pyplot as plt
import cv2

from Camera import Camera 

# This function plots given cameras (i.e., rvecs, tvecs, cameraMats, distCoeffs, imgSizes) and object points (i.e., objPoints)
# and create a 3D plot of the cameras and the object points.
# Each of rvecs, tvecs, cameraMats, and distCoeffs is a list, where the list length is the number of cameras.
# objPoints is a nPoints x 3 numpy float array, where nPoints is the number of points.
# Each camera is represented by a 3D cone, where the cone apex is the camera position and the cone axis is 
# the camera direction.
# The object points are represented by small spheres.
def plotCameras3d(imgSizes, rvecs, tvecs, cameraMats, distCoeffs, objPoints=None):
    # number of cameras
    nCameras = len(rvecs)
    if nCameras != len(tvecs) or nCameras != len(cameraMats) or nCameras != len(distCoeffs):
        print("# plotCameras3d(): Error: the number of cameras is not consistent.")
        return
    # number of points
    if objPoints is None or objPoints.shape[0] <= 0:
        nPoints = 0
    elif objPoints.shape[1] != 3:
        print("# plotCameras3d(): Error: the object points should be a nPoints x 3 numpy float array." 
              " But its shape is: ", objPoints.shape)
        return
    else:
        nPoints = objPoints.shape[0]
    # create a 3D plot. The title of the window is "3D plot of cameras and object points"
    fig = plt.figure()
    fig.canvas.manager.set_window_title('3D Plot of Cameras')
    ax = fig.add_subplot(111, projection='3d')
    # estimate the scale of the 3D space
    # set proper limits so that allCameraPos is inside the plot and the aspect ratio is the same
    allVertices = []
    #   append all object points into allVertices
    cameras = []
    if objPoints is not None and type(objPoints) == np.ndarray \
        and objPoints.shape[0] > 0 and objPoints.shape[1] == 3:
        for i in range(objPoints.shape[0]):
            allVertices.append(objPoints[i])
    if nPoints > 0:
        for i in range(nCameras):
            # get the camera position and direction
            imgSize = imgSizes[i]
            rvec = rvecs[i]
            tvec = tvecs[i]
            cameraMat = cameraMats[i]
            distCoeff = distCoeffs[i]
            # create a camera object
            thisCam = Camera(imgSize, rvec, tvec, cameraMat, distCoeff)
            cameras.append(thisCam)
            allVertices.append(thisCam.campos().flatten())
    # convert allVertices to a (nPoints+nCameras) x 3 numpy array
    allVertices = np.array(allVertices).reshape(-1, 3) 
    #     find all vertices of the object points and cameras
    max_x = np.max(allVertices[:, 0])
    min_x = np.min(allVertices[:, 0])
    max_y = np.max(allVertices[:, 1])
    min_y = np.min(allVertices[:, 1])
    max_z = np.max(allVertices[:, 2])
    min_z = np.min(allVertices[:, 2])
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    max_range = np.max([max_x - min_x, max_y - min_y, max_z - min_z]) / 2
    # plot the object points
    if nPoints > 0:
        ax.scatter(objPoints[:,0], objPoints[:,1], objPoints[:,2], c='r', marker='o')
    # plot the cameras
    for i in range(nCameras):
        thisCam = cameras[i]
        ax = thisCam.plotCameraPyramid(color='green', alpha=0.4, axes=ax, pyramid_height=0.1*max_range)
    # set proper limits so that allCameraPos is inside the plot and the aspect ratio is the same
    ax.set_xlim([center_x - max_range, center_x + max_range])
    ax.set_ylim([center_y - max_range, center_y + max_range])
    ax.set_zlim([center_z - max_range, center_z + max_range])
    # show the plot
    plt.show()
    pass

# plot cameras in a matplotlib 3d axes. Each camera is represented by a 3d cone. 
# This function uses Camera class in Camera.py to plot. 
# Input:
#    rvecs: a list of n_cameras elements, where each element is a 3x1 numpy array.
#    tvecs: a list of n_cameras elements, where each element is a 3x1 numpy array.
#    cmats: a list of n_cameras elements, where each element is a 3x3 numpy array.
#    dvecs: a list of n_cameras elements, where each element is a 1x5 numpy array.
#    h_cone: height of each cone. The default is 0.5.
#    axes: the axes of the plot. The default is None. If None, a new figure will be created.
#    consistentScale: if True, the scale of the plot is consistent. The default is True.
# Output:
#    None
def plot_cameras(rvecs, tvecs, cmats, dvecs, img_sizes, h_cone=1, axes=None, 
                 xlim=None, ylim=None, zlim=None, 
                 equal_aspect_ratio=True):
    # create a new figure if theAxes is None
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
    # plot the cameras
    # get camera positions and store at camspos, a n_cameras x 3 numpy array
    camspos = np.zeros((len(rvecs), 3), dtype=float)
    for i in range(len(rvecs)):
        theCam = Camera()
        theCam.rvec = rvecs[i]
        theCam.tvec = tvecs[i]
        theCam.cmat = cmats[i]
        theCam.dvec = dvecs[i]
        theCam.imgSize = img_sizes[i]
        theCam.plotCameraPyramid(color='green', alpha=0.5, axes=axes, pyramid_height=h_cone)
        # set camera position to camspos
        camspos[i] = theCam.campos().flatten()
    # label the axes
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    # set the xlim, ylim, and zlim so that 
    if xlim is None:
        xlim = [np.min(camspos[:, 0]), np.max(camspos[:, 0])]
    if ylim is None:
        ylim = [np.min(camspos[:, 1]), np.max(camspos[:, 1])]
    if zlim is None:
        zlim = [np.min(camspos[:, 2]), np.max(camspos[:, 2])]
    # adjust xlim, ylim, zlim, so that the x/y/z scales are the same
    if equal_aspect_ratio is True:
        xrange = abs(xlim[1] - xlim[0])
        yrange = abs(ylim[1] - ylim[0])
        zrange = abs(zlim[1] - zlim[0])
        max_range = max([xrange, yrange, zrange])
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_zlim(zlim)
        axes.set_xlim3d([np.mean(xlim) - max_range / 2, np.mean(xlim) + max_range / 2])
        axes.set_ylim3d([np.mean(ylim) - max_range / 2, np.mean(ylim) + max_range / 2])
        axes.set_zlim3d([np.mean(zlim) - max_range / 2, np.mean(zlim) + max_range / 2])

    return axes



def unitTest01():
    #  create truePos3Ds: randomly generate several 3D points in a unit sphere (nPoints) and store 
    #     them in a list (truePos3Ds) which is a nPoints x 3 numpy array.
    nPoints = 10
    # generate random 3D points in a unit sphere
    truePos3Ds = np.random.randn(nPoints, 3)
    # normalize the points
    truePos3Ds = truePos3Ds / np.linalg.norm(truePos3Ds, axis=1)[:, None]
    # multiply by a random scale
    truePos3Ds = truePos3Ds * np.random.rand(nPoints)[:, None]
    # print data of each point in this form: (x, y, z). Distance to origin: d
    print("*"*60+"# Step 1: create truePos3Ds, random 3D points in a unit sphere.")
    for i in range(nPoints):
        print("# Point %d: (%f, %f, %f). Distance to origin: %f" % (i, truePos3Ds[i, 0], truePos3Ds[i, 1], truePos3Ds[i, 2], np.linalg.norm(truePos3Ds[i])))

#     generate rvecs, tvecs, cameraMats, and distCoeffs: generate several (nCameras) camera 
#     parameters (rvecs, tvecs, cameraMats, distCoeffs) that aims about the center of the sphere, 
#     each of them is a list of nCameras elements.
#     Each rvecs element is a 3x1 numpy array, each tvecs element is a 3x1 numpy array,
#     each cameraMats element is a 3x3 numpy array, and each distCoeffs element is a 1x5 or 
#     1x8 numpy array.
    from rvecTvecFromPosAim import rvecTvecFromPosAim
    nCameras = 4
    rvecs = []
    tvecs = []
    cameraMats = []
    distCoeffs = []
    imgSize = (1920, 1080)
    imgSizes = [imgSize] * nCameras
    fov_h = 45 * np.pi / 180. # radians
    fov_v = fov_h * imgSize[1] / imgSize[0]
    fx = imgSize[0] / (2 * np.tan(fov_h / 2))
    fy = imgSize[1] / (2 * np.tan(fov_v / 2))
    cx = (imgSize[0]-1) / 2
    cy = (imgSize[1]-1) / 2
    cameraMat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cameraMats = [cameraMat] * nCameras
    # cameras are randomly but must be placed on the x-y plane, and the distance to the origin is 3.0
    # the camera positions is generated and stored at cameraPos which is a nCameras x 3 numpy array
    cameraPos = np.random.randn(nCameras, 3)
    # set camera position height (z) a random number between +/- 0.1
    cameraPos[:, 2] = np.random.rand(nCameras) * 0.2 - 0.1
    cameraPos = cameraPos / np.linalg.norm(cameraPos, axis=1)[:, None] * 3.0
    # calculates rvecs and tvecs from cameraPos
    for i in range(nCameras):
        rvec, tvec = rvecTvecFromPosAim(cameraPos[i], np.array([0, 0, 0]))
        rvecs.append(rvec)
        tvecs.append(tvec)
        k1 = np.random.randn() * 0.01
        k2 = np.random.randn() * 0.01
        p1 = np.random.randn() * 0.01
        p2 = np.random.randn() * 0.01
        k3 = np.random.randn() * 0.01
        distCoeffs.append(np.array([k1,k2,p1,p2,k2,k3]))
    print("*"*60+"\n# Step 2: generate camera parameters.")

    # print each camera parameters in this form fx fy cx cy k1 k2 p1 p2 k3
    for i in range(nCameras):
        print("# Camera %d: fx=%f, fy=%f, cx=%f, cy=%f, k1=%f, k2=%f, p1=%f, p2=%f, k3=%f" % (i, cameraMats[i][0, 0], cameraMats[i][1, 1], cameraMats[i][0, 2], cameraMats[i][1, 2], distCoeffs[i][0], distCoeffs[i][1], distCoeffs[i][2], distCoeffs[i][3], distCoeffs[i][4]))
        print("# Camera %d: rvec=%s, tvec=%s" % (i, rvecs[i].T, tvecs[i].T))
    # plot cameras
    plotCameras3d(imgSizes, rvecs, tvecs, cameraMats, distCoeffs, objPoints=truePos3Ds)
    pass


# test program of plotCameras3d
if __name__ == '__main__':
    # unitTest01()
    unitTest01()

# Done
