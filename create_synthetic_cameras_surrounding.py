import numpy as np
import cv2
from Camera import Camera

# This function creates synthetic data for multiple cameras that surround a point.
# All cameras aim the given point and are located on a circle around the point.
# This function returns rvecs, tvecs, cmats, dvecs.
# Input:
#    n_cameras: the number of cameras. The default is 4.
#    center: the center of the object points. The default is np.array([0.0, 0.0, 0.0]).
#    img_size: image size of cameras. The default is (1024, 768).
#    fov_x: the field of view of cameras along x axis in degrees. The default is 60.0.
#    dvec: the distortion coefficients of cameras. The default is (0.1, 0.05, 0.05, 0.0, 0.0).
#    radius_range: the range of radius of the circle where cameras are located. The default is (6.0, 8.0).
#    height_range: the range of height of the circle where cameras are located. The default is (2.0, 5.0).
# Output:
#    rvecs: a list of n_cameras elements, where each element is a 3x1 numpy array.
#    tvecs: a list of n_cameras elements, where each element is a 3x1 numpy array.
#    cmats: a list of n_cameras elements, where each element is a 3x3 numpy array.
#    dvecs: a list of n_cameras elements, where each element is a 1x5 numpy array.
#    img_sizes: a list of n_cameras elements, where each element is a tuple of (width, height).
def create_synthetic_cameras_surrounding(
    n_cameras=4, 
    center=np.array([0.0, 0.0, 3.0]),
    img_size=(1024, 768),
    fov_x=60.0,
    dvec=(0.1, 0.05, 0.05, 0.0, 0.0),
    radius_range=(6.0, 8.0),
    height_range=(2.0, 5.0) 
    ):
    # create cameras: randomly generate several (nCameras) camera (Camera objects):
    # every camera aims the center, and the camera position is around the center. 
    # The distance to the center (ignoring the z) is the range of radius (radius_range).
    # The height of cameras (z) is the range of height (height_range).
    # cameras = [], a list of Camera objects.
    cameras = []
    rvecs = []
    tvecs = []
    cmats = []
    dvecs = []
    img_sizes = []
    for i in range(n_cameras):
        # the camera height (z) is a random between height_range[0] and height_range[1]
        camz = np.random.rand() * (height_range[1] - height_range[0]) + height_range[0]
        # the theta is a random number between 0 and 2 * np.pi
        theta = np.random.rand() * 2 * np.pi
        # the radius is a random number between radius_range[0] and radius_range[1]
        radius = np.random.rand() * (radius_range[1] - radius_range[0]) + radius_range[0]
        camx = center[0] + radius * np.cos(theta)
        camy = center[1] + radius * np.sin(theta)
        theta = np.random.rand() * 2 * np.pi
        camera_position = np.array([camx, camy, camz])
        # append an empty Camera into the list
        cameras.append(Camera())
        cameras[i].setRvecTvecByPosAim(camera_position, center)
        cameras[i].setCmatByImgsizeFovs(img_size, fov_x)
        cameras[i].dvec = np.array(dvec, dtype=float).reshape(-1, 1)
        cameras[i].imgSize = img_size
        # append rvec, tvec, cmat, and dvec
        rvecs.append(cameras[i].rvec)
        tvecs.append(cameras[i].tvec)
        cmats.append(cameras[i].cmat)
        dvecs.append(cameras[i].dvec)
        img_sizes.append(cameras[i].imgSize)
    return rvecs, tvecs, cmats, dvecs, img_sizes


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

# test the function
if __name__ == '__main__':
    # create a tkinter window and make title as "Test create_synthetic_cameras_surrounding"
    # with a text entry for the number of cameras, default is "4"
    # with a text entry for the center of the point range, default is "(0.0, 0.0, 0.0)"
    # with a text entry for the image size, default is "(1024, 768)"
    # with a text entry for the camera field of view in horizontal (in degrees), default is "60.0"
    # with a text entry for the distortion coefficients of cameras, default is "(0.1, 0.05, 0.05, 0.0, 0.0)"
    # with a text entry for the radius of the circle where cameras are located, default is "3.0"
    # with a button to generate the synthetic data by calling create_synthetic_cameras_surrounding and plot_cameras().
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import tkinter as tk
    # create a tkinter window and set the width of the window as 800 and height as 600
    root = tk.Tk()
    root.geometry("800x600")
#    root.withdraw()
    root.title("Test create_synthetic_cameras_surrounding" + " "*20 + "githut copilot generated under supervision of Vince Yang")
    # create a text entry for the number of cameras
    label = tk.Label(root, text="Number of cameras:")
    label.pack()
    nCamerasEntry = tk.Entry(root)
    nCamerasEntry.pack()
    nCamerasEntry.insert(0, "4")
    # create a text entry for the center of the point range
    label = tk.Label(root, text="Center of the point range (x, y, z):")
    label.pack()
    centerOfPointRangeEntry = tk.Entry(root)
    centerOfPointRangeEntry.pack()
    centerOfPointRangeEntry.insert(0, "(0.0, 0.0, 3.0)")
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
    label = tk.Label(root, text="Distortion coefficients:")
    label.pack()
    dvecEntry = tk.Entry(root)
    dvecEntry.pack()
    dvecEntry.insert(0, "(0.1, 0.05, 0.05, 0.0, 0.0)")
    # create a text entry for the range of the radius of the circle where cameras are located
    label = tk.Label(root, text="Range of radius of circle where cameras are located:")
    label.pack()
    radiusEntry = tk.Entry(root)
    radiusEntry.pack()
    radiusEntry.insert(0, "(6.0, 8.0)")
    # create a text entry for the range of the height of the cameras
    label = tk.Label(root, text="Range of height of the cameras:")
    label.pack()
    heightEntry = tk.Entry(root)
    heightEntry.pack()
    heightEntry.insert(0, "(2.0, 5.0)")
    # create a text entry for xlim of the plot
    label = tk.Label(root, text="xlim of the plot:")
    label.pack()
    xlimEntry = tk.Entry(root)
    xlimEntry.pack()
    xlimEntry.insert(0, "(-10, 10)")
    # create a text entry for ylim of the plot
    label = tk.Label(root, text="ylim of the plot:")
    label.pack()
    ylimEntry = tk.Entry(root)
    ylimEntry.pack()
    ylimEntry.insert(0, "(-10, 10)")
    # create a text entry for zlim of the plot
    label = tk.Label(root, text="zlim of the plot:")
    label.pack()
    zlimEntry = tk.Entry(root)
    zlimEntry.pack()
    zlimEntry.insert(0, "(-10, 10)")
    # create a button to generate the synthetic data
    def button_event_plot3d_test_synthetic_cameras_surrounding():
        n_cameras = int(nCamerasEntry.get())
        center = eval(centerOfPointRangeEntry.get())
        img_size = eval(imgSizeEntry.get())
        fov_x = float(camFovxEntry.get())
        dvec = eval(dvecEntry.get())
        radius_range = eval(radiusEntry.get())
        height_range = eval(heightEntry.get())
        rvecs, tvecs, cmats, dvecs, img_sizes = create_synthetic_cameras_surrounding(
            n_cameras, center, img_size, fov_x, dvec, radius_range, height_range)
        # plot the cameras in 3D
        try:
            xlim = eval(xlimEntry.get())
            if len(xlim) != 2:
                xlim = None
        except:
            xlim = None
        try:
            ylim = eval(ylimEntry.get())
            if len(ylim) != 2:
                ylim = None
        except:
            ylim = None
        try:
            zlim = eval(zlimEntry.get())
            if len(zlim) != 2:
                zlim = None
        except:
            zlim = None
        plot_cameras(rvecs, tvecs, cmats, dvecs, img_sizes, h_cone=1.0, axes=None, 
                     xlim=xlim, ylim=ylim, zlim=zlim, equal_aspect_ratio=True)
        plt.show()
    button = tk.Button(root, text="Plot Cameras in 3D", command=button_event_plot3d_test_synthetic_cameras_surrounding)
    button.pack()
    root.mainloop()

