import cv2
import numpy as np
from Camera import Camera 
from create_synthetic_3d_points import create_synthetic_3d_points, plot_3dPoints_and_sphere



# This function creates synthetic data for multiple cameras and multiple object points.
# The cameras are static and the object points are static.
# This function returns rvecs, tvecs, cmats, dvecs, objPoints, imgPointss.
# Input:
#    nCameras: the number of cameras. The default is 4.
#    nPoints: the number of object points. The default is 60. 
#    centerOfPointRange: the center of the object points. The default is np.array([0.0, 0.0, 0.0]).
#    
def createSyntheticDataMultCamMultiPointsStatic(
    nCameras=4, 
    nPoints=6, 
    centerOfPointRange=np.array([0.0, 0.0, 0.0]),
    radiusOfPointRange=1.0,
    imgSize=(1024, 768),
    camFovx = 60.0,
    distCoeffs=(0.1, 0.05, 0.05, 0.0, 0.0)):

    #  1. create truePos3Ds: randomly generate several 3D points (nPoints) 
    #     in a sphere, which center is at CenterOfPointRange, and radius 
    #     is radiusOfPointRange, and store 
    #     them in a list (truePos3Ds) which is a nPoints x 3 numpy array.
    #truePos3Ds = createSynthetic3dPoints(center=centerOfPointRange, radius=radiusOfPointRange, nPoints=nPoints)
    truePos3Ds = create_synthetic_3d_points(center=centerOfPointRange, radius=radiusOfPointRange, nPoints=nPoints)
    # print data of each point in this form: (x, y, z). Distance to centerOfPointRange: d
    print("*"*60+"# Step 1: create truePos3Ds, random 3D points in a unit sphere.")
    for i in range(nPoints):
        print("# Point %d: (%f, %f, %f). Distance to origin: %f" % (i, truePos3Ds[i, 0], truePos3Ds[i, 1], truePos3Ds[i, 2],
                                                                     np.linalg.norm(truePos3Ds[i] - centerOfPointRange)))
    #  2. create cameras: randomly generate several (nCameras) camera (Camera objects):
    #     every camera aims the center of the sphere (centerOfPointRange), and the camera position is
    #     around the sphere. The distance to the center is at least 3 times the radius of the sphere.
    #     cameras = [], a list of Camera objects.
    cameras = []
    for i in range(nCameras):
        # the camera height (z) is the centerOfPointRange[2] + a random number between +/- 0.5 * radiusOfPointRange
        # the camera position x is the centerOfPointRange[0] + 3 * radiusOfPointRange * np.cos(random angle between 0 and 2 * np.pi)
        # the camera position y is the centerOfPointRange[0] + 3 * radiusOfPointRange * np.sin(random angle between 0 and 2 * np.pi)
        # the camera position is np.array([camx, camy, camz])
        camz = centerOfPointRange[2] + (np.random.rand() - 0.5) * radiusOfPointRange
        camx = centerOfPointRange[0] + 3 * radiusOfPointRange * np.cos(np.random.rand() * 2 * np.pi)
        camy = centerOfPointRange[1] + 3 * radiusOfPointRange * np.sin(np.random.rand() * 2 * np.pi)
        cameraPos = np.array([camx, camy, camz])
        # append an empty Camera into the list
        cameras.append(Camera())
        cameras[i].imgSize = (imgSize[0], imgSize[1])
        cameras[i].setRvecTvecByPosAim(cameraPos, centerOfPointRange)
        cameras[i].setCmatByImgsizeFovs(imgSize, camFovx)
        cameras[i].dvec = np.array(distCoeffs, dtype=float).reshape(-1, 1)
    # print the camera infomation on the screen for verification
    for i in range(nCameras):
        print("*"*60+"# Step 2: create cameras.")
        print("  Camera %d:" % i, cameras[i])

# test the function
def test_createSyntheticDataMultCamMultiPointsStatic():
    # create a tkinter window and make title as "Test createSyntheticDataMultCamMultiPointsStatic
    # with a text entry for the number of cameras, default is "4"
    # with a text entry for the number of points, default is "6"
    # with a text entry for the center of the point range, default is "(0.0, 0.0, 0.0)"
    # with a text entry for the radius of the point range, default is "1.0"
    # with a text entry for the image size, default is "(1024, 768)"
    # with a text entry for the camera field of view in horizontal (in degrees), default is "60.0"
    # with a text entry for the distortion coefficients, default is "(0.1, 0.05, 0.05, 0.0, 0.0)"
    # with a button to generate the synthetic data by calling createSyntheticDataMultCamMultiPointsStatic
    #      then plots them by calling plot_3dPoints_and_sphere()
    import tkinter as tk
    root = tk.Tk()
#    root.withdraw()
    root.title("Test createSyntheticDataMultCamMultiPointsStatic" + " "*20 + "githut copilot generated under supervision of Vince Yang")
    # create a text entry for the number of cameras
    label = tk.Label(root, text="Number of cameras:")
    label.pack()
    nCamerasEntry = tk.Entry(root)
    nCamerasEntry.pack()
    nCamerasEntry.insert(0, "4")
    # create a text entry for the number of points
    label = tk.Label(root, text="Number of points:")
    label.pack()
    nPointsEntry = tk.Entry(root)
    nPointsEntry.pack()
    nPointsEntry.insert(0, "6")
    # create a text entry for the center of the point range
    label = tk.Label(root, text="Center of the point range (x, y, z):")
    label.pack()
    centerOfPointRangeEntry = tk.Entry(root)
    centerOfPointRangeEntry.pack()
    centerOfPointRangeEntry.insert(0, "(0.0, 0.0, 0.0)")
    # create a text entry for the radius of the point range
    label = tk.Label(root, text="Radius of the point range:")
    label.pack()
    radiusOfPointRangeEntry = tk.Entry(root)
    radiusOfPointRangeEntry.pack()
    radiusOfPointRangeEntry.insert(0, "1.0")
    # create a text entry for the image size
    label = tk.Label(root, text="Image size (width, height):")
    label.pack()
    imgSizeEntry = tk.Entry(root)
    imgSizeEntry.pack()
    imgSizeEntry.insert(0, "(1024, 768)")
    # create a text entry for the camera field of view in horizontal (in degrees)
    label = tk.Label(root, text="Camera field of view in horizontal (in degrees):")
    label.pack()
    camFovxEntry = tk.Entry(root)
    camFovxEntry.pack()
    camFovxEntry.insert(0, "60.0")
    # create a text entry for the distortion coefficients
    label = tk.Label(root, text="Distortion coefficients (k1, k2, p1, p2, k3):")
    label.pack()
    distCoeffsEntry = tk.Entry(root)
    distCoeffsEntry.pack()
    distCoeffsEntry.insert(0, "(0.1, 0.05, 0.05, 0.0, 0.0)")
    # create a button to generate the synthetic data

    def createAndPlot():
        createSyntheticDataMultCamMultiPointsStatic(
            nCameras=int(nCamerasEntry.get()), 
            nPoints=int(nPointsEntry.get()), 
            centerOfPointRange=eval(centerOfPointRangeEntry.get()),
            radiusOfPointRange=float(radiusOfPointRangeEntry.get()),
            imgSize=eval(imgSizeEntry.get()),
            camFovx=float(camFovxEntry.get()),
            distCoeffs=eval(distCoeffsEntry.get()))
        plot_3dPoints_and_sphere(truePos3Ds, centerOfPointRange, radiusOfPointRange)


    button = tk.Button(root, text="Generate synthetic data", command=lambda: createSyntheticDataMultCamMultiPointsStatic(
        nCameras=int(nCamerasEntry.get()), 
        nPoints=int(nPointsEntry.get()), 
        centerOfPointRange=eval(centerOfPointRangeEntry.get()),
        radiusOfPointRange=float(radiusOfPointRangeEntry.get()),
        imgSize=eval(imgSizeEntry.get()),
        camFovx=float(camFovxEntry.get()),
        distCoeffs=eval(distCoeffsEntry.get())))
    button.pack()
    # show the root window
    root.mainloop()
    pass
    pass



# main
if __name__ == '__main__':
    test_createSyntheticDataMultCamMultiPointsStatic()