import cv2
import numpy as np
from math import *
from triangulatePoints2 import triangulatePoints2
from Camera import Camera

# function multi_camera_triangulation: given camera parameters of multiple cameras (i.e., rvecs, 
# tvecs, cmats, dvecs, each of them is a list, where the list length is the number
# of cameras), image points of multiple points 
# (i.e., img_pointss, a list of 2D points in multiple cameras)
# a function that calculates the 3D position of points 
# from the 2D positions of the point in multiple cameras.
# Input arguments:
#  rvecs: a list of rotation vectors of the cameras. Each element is a 3x1 numpy array.
#       That is, rvecs[i] is the rotation vector of the i-th camera.
#       The length of rvecs (i.e., len(rvecs)) should be the number of cameras.
#       The rvec, tvec, cmat, and dvec is normally obtained from camera calibration, 
#       such as cv2.calibrateCamera(), or similar functions, e.g., cv2.solvePnP(). 
#  tvecs: a list of translation vectors of the cameras. Each element is a 3x1 numpy array.
#       That is, tvecs[i] is the translation vector of the i-th camera.
#       The length of tvecs (i.e., len(tvecs)) should be the number of cameras.
#  cmats: a list of camera matrices of the cameras. Each element is a 3x3 numpy array.
#       That is, cmats[i] is the camera matrix of the i-th camera.
#       The length of cmats (i.e., len(cmats)) should be the number of cameras.
#  dvecs: a list of distortion coefficients of the cameras. Each element is a 1x4, 1x5, 
#       or 1x8 numpy array.
#       The contents of a dvec is k1, k2, p1, p2, k3, k4, k5, k6
#       The length of dvecs (i.e., len(dvecs)) should be the number of cameras.
#  img_pointss: a list of image points of the points in multiple
#       cameras. img_pointss is a list of n_cameras elements, where each element is a
#       n_points x 2 numpy array. That is, img_pointss[i] is the image points of the 
#       i-th camera. n_points is the number of points. The number of points in each camera
#       should be the same. If a certain point is not detected in a camera, the point should
#       be marked as [np.nan, np.nan].
# Output:
#  triangulated_points: the 3D position of the points from the 2D positions of the point 
#       in multiple cameras. The dimension of triangulated_pos3ds is n_points x 3. 
#       That is, triangulated_pos3ds[i,:] is the 3D position of the i-th point.
#       If the triangulation is failed, for example, less than two cameras can see it, 
#       the 3D position of the point is [np.nan, np.nan, np.nan]
#  projected_pointss: a list of the 2D position of the 3D points in each camera. The 
#       dimension of projected_points is the same as the input img_pointss.
#       That is, projected_pointss[i] is the 2D position of the 3D points in the i-th camera.
#       Even if a point is not seen by a camera, the point is projected to the image plane.
#       If the triangulation is failed, the projected 2D position of the point is [np.nan, np.nan].
# Example1:
#       rvecs = [np.array([0, 0, 0]), np.array([0, 0, 0])]  
#       tvecs = [np.array([0, 0, 0]), np.array([0, 0, 0])]  
#       cmats = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
#       dvecs = [np.array([0, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 0])]
#       img_pointss = [np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 0], [1, 1], [2, 2]])]
#       # You need to replace the above rvecs, tvecs, cmats, and dvecs with the real values.
#       triangulated_points, projected_pointss = multi_camera_triangulation(rvecs, tvecs, 
#           cmats, dvecs, img_pointss)
def multi_camera_triangulation(rvecs, tvecs, cmats, dvecs, img_pointss):
    # number of cameras
    n_cameras = len(rvecs)
    if n_cameras < 2:
        print("# multi_camera_triangulation(): Error: the number of cameras is less than 2.")
        return None
    if n_cameras != len(tvecs) or n_cameras != len(cmats) or n_cameras != len(dvecs) or n_cameras != len(img_pointss):
        print("# multi_camera_triangulation(): Error: the number of cameras is not consistent.")
        print("# multi_camera_triangulation(): len(rvecs) is %d." % len(rvecs))
        print("# multi_camera_triangulation(): len(tvecs) is %d." % len(tvecs))
        print("# multi_camera_triangulation(): len(cmats) is %d." % len(cmats))
        print("# multi_camera_triangulation(): len(dvecs) is %d." % len(dvecs))
        print("# multi_camera_triangulation(): They should be the same. Otherwise the function cannot work.")
        return None
    # number of points
    n_points = len(img_pointss[0])
    if n_points != len(img_pointss[1]):
        print("# multi_camera_triangulation(): Error: the number of points is not consistent.")
        for i in range(n_cameras):
            print("# multi_camera_triangulation(): len(img_pointss[%d]) is %d." % (i, len(img_pointss[i])))
        print("# multi_camera_triangulation(): They should be the same. Otherwise the function cannot work.")
        return None
    # determine all possible composition of camera pair among all cameras
    # for example, if there are four cameras, generate cam_pairs as follows:
    # cam_pairs = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    cam_pairs = []
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            cam_pairs.append([i, j])
    ncam_pairs = len(cam_pairs)
    # calculate the 3D position of the points from the 2D positions of the point in multiple cameras
    # by using triangulatePoints() function in cv2
    # triangulated_pos3ds: the 3D position of the points from the 2D positions of the point in multiple cameras
    # the dimension of triangulated_pos3ds is n_points x ncam_pairs x 3
    # i.e., triangulated_pos3ds[i,j,:] is the 3D position of the i-th point from the j-th camera pair
    triangulated_pos3ds_all_pairs = np.nan * np.ones((n_points, ncam_pairs, 3))
    for j in range(ncam_pairs):
        # get the camera pair
        cam_pair = cam_pairs[j]
        # get the image points of the point in the camera pair
        img_points_pair = [img_pointss[cam_pair[0]], img_pointss[cam_pair[1]]]
        # calculate the 3D position of the point from the 2D positions of the point in the camera pair
        # by using triangulatePoints2() function
        objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2 =\
            triangulatePoints2(cmats[cam_pair[0]], dvecs[cam_pair[0]], 
                               rvecs[cam_pair[0]], tvecs[cam_pair[0]], 
                               cmats[cam_pair[1]], dvecs[cam_pair[1]], 
                               rvecs[cam_pair[1]], tvecs[cam_pair[1]], 
                               img_points_pair[0], img_points_pair[1])
            # store the result
        triangulated_pos3ds_all_pairs[:,j,:] = objPoints
    # calculate the average of each point from all camera pairs
#    triangulated_pos3ds_best = np.nanmean(triangulated_pos3ds_all_pairs, axis=1)
    # calculate the centroid of the inliers (by KNN) of each point from all camera pairs
    from inliers_3d import inliers_centroid_3d
    triangulated_pos3ds_best = np.nan * np.ones((n_points, 3))
    for i in range(n_points):
        inliers_best = inliers_centroid_3d(triangulated_pos3ds_all_pairs[i,:,:].reshape(-1,3), k=5, percentile=50)
        triangulated_pos3ds_best[i,:] = inliers_best
    #
    triangulated_points = triangulated_pos3ds_best
    # project the triangulated 3D points to the image plane of each camera
    # projected_pointss: a list of the 2D position of the 3D points in each camera
    projected_pointss = []
    for i in range(n_cameras):
        projected_points = cv2.projectPoints(triangulated_points, rvecs[i], tvecs[i], cmats[i], dvecs[i])[0].reshape(-1, 2)
        projected_pointss.append(projected_points)
    return triangulated_points, projected_pointss


from rvecTvecFromPosAim import rvecTvecFromPosAim 
# a unit test of multi_camera_triangulation
# This function does:
#  1. create truePos3Ds: randomly generate several 3D points in a unit sphere (n_points) and store 
#     them in a list (truePos3Ds) which is a n_points x 3 numpy array.
#  2. generate rvecs, tvecs, cmats, and dvecs: generate several (n_cameras) camera 
#     parameters (rvecs, tvecs, cmats, dvecs) that aims about the center of the sphere, 
#     each of them is a list of n_cameras elements.
#     Each rvecs element is a 3x1 numpy array, each tvecs element is a 3x1 numpy array,
#     each cmats element is a 3x3 numpy array, and each dvecs element is a 1x5 or 
#     1x8 numpy array.
#  3. calculate img_points: calculate the image points of the 3D points in each camera, and store them in a list 
#     (img_points). img_points is a list of n_cameras elements, where each element is a 
#     n_points x 2 numpy array.
#  4. calculate triangulated_pos3ds: calculate the 3D position of the points from the 2D positions of the point in multiple 
#     cameras by calling multi_camera_triangulation. Store the results in triangulated_pos3ds, 
#     which is a nPoint x 3 numpy array.
#  5. calculate errorPos3Ds: the error between the truePos3Ds and the triangulated_pos3ds. Print it. 
#  6. calculate projectPos2Ds: project the triangulated_pos3ds to projectPos2Ds by calling projectPoints. 
#     Store the results in projectPos2Ds, which is a list of n_cameras elements, where each element 
#     is a n_points x 2 numpy array.
#  7. calculate errorPos2Ds: calculate the error between the img_points and the projectPos2Ds. Print it.



# unit test of multi_camera_triangulation
def unit_test_v00():

    #  1. create truePos3Ds: randomly generate several 3D points in a unit sphere (n_points) and store 
    #     them in a list (truePos3Ds) which is a n_points x 3 numpy array.
    n_points = 5
    # generate random 3D points in a unit sphere
    truePos3Ds = np.random.randn(n_points, 3)
    # normalize the points
    truePos3Ds = truePos3Ds / np.linalg.norm(truePos3Ds, axis=1)[:, None]
    # multiply by a random scale
    truePos3Ds = truePos3Ds * np.random.rand(n_points)[:, None]
    # print data of each point in this form: (x, y, z). Distance to origin: d
    print("*"*60+"# Step 1: create truePos3Ds, random 3D points in a unit sphere.")
    for i in range(n_points):
        print("# Point %d: (%f, %f, %f). Distance to origin: %f" % (i, truePos3Ds[i, 0], truePos3Ds[i, 1], truePos3Ds[i, 2], np.linalg.norm(truePos3Ds[i])))
    
#  2. generate rvecs, tvecs, cmats, and dvecs: generate several (n_cameras) camera 
#     parameters (rvecs, tvecs, cmats, dvecs) that aims about the center of the sphere, 
#     each of them is a list of n_cameras elements.
#     Each rvecs element is a 3x1 numpy array, each tvecs element is a 3x1 numpy array,
#     each cmats element is a 3x3 numpy array, and each dvecs element is a 1x5 or 
#     1x8 numpy array.
    from rvecTvecFromPosAim import rvecTvecFromPosAim
    n_cameras = 4
    rvecs = []
    tvecs = []
    cmats = []
    dvecs = []
    imgSize = (1920, 1080)
    fov_h = 100 * pi / 180. # radians
    fov_v = fov_h * imgSize[1] / imgSize[0]
    fx = imgSize[0] / (2 * np.tan(fov_h / 2))
    fy = imgSize[1] / (2 * np.tan(fov_v / 2))
    cx = (imgSize[0]-1) / 2
    cy = (imgSize[1]-1) / 2
    cameraMat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cmats = [cameraMat] * n_cameras
    # cameras are randomly but must be placed on the x-y plane, and the distance to the origin is 3.0
    # the camera positions is generated and stored at cameraPos which is a n_cameras x 3 numpy array
    cameraPos = np.random.randn(n_cameras, 3)
    cameraPos[:, 2] = 0
    cameraPos = cameraPos / np.linalg.norm(cameraPos, axis=1)[:, None] * 3.0
    # calculates rvecs and tvecs from cameraPos
    for i in range(n_cameras):
        rvec, tvec = rvecTvecFromPosAim(cameraPos[i], np.array([0, 0, 0]))
        rvecs.append(rvec)
        tvecs.append(tvec)
        k1 = np.random.randn() * 0.01
        k2 = np.random.randn() * 0.01
        p1 = np.random.randn() * 0.01
        p2 = np.random.randn() * 0.01
        k3 = np.random.randn() * 0.01
        k4 = np.random.randn() * 0.01
        k5 = np.random.randn() * 0.01
        k6 = np.random.randn() * 0.01
        dvecs.append(np.array([k1,k2,p1,k2,k3,k4,k5,k6]))
    print("*"*60+"\n# Step 2: generate camera parameters.")
    # print each camera parameters in this form fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 
    for i in range(n_cameras):
        print("# Camera %d: fx=%f, fy=%f, cx=%f, cy=%f, k1=%f, k2=%f, p1=%f, p2=%f, k3=%f, k4=%f, k5=%f, k6=%f" % 
              (i, cmats[i][0, 0], cmats[i][1, 1], cmats[i][0, 2], cmats[i][1, 2], dvecs[i][0], dvecs[i][1], 
               dvecs[i][2], dvecs[i][3], dvecs[i][4], dvecs[i][5], dvecs[i][6], dvecs[i][7]))
        print("# Camera %d: rvec=%s, tvec=%s" % (i, rvecs[i].T, tvecs[i].T))

#  3. calculate img_points: calculate the image points of the 3D points in each camera, and store them in a list 
#     (img_points). img_points is a list of n_cameras elements, where each element is a 
#     n_points x 2 numpy array.
    img_points = []
    for i in range(n_cameras):
        img_points.append(cv2.projectPoints(truePos3Ds, rvecs[i], tvecs[i], 
                                           cmats[i], dvecs[i])[0].reshape(-1, 2))
    print("*"*60+"\n# Step 3: calculate img_points.")
    # print the image points of each camera
    for i in range(n_cameras):
        print("# Camera %d:" % i)
        for j in range(n_points):
            print("# Point %d: (%f, %f)" % (j, img_points[i][j, 0], img_points[i][j, 1]))
#  4. calculate triangulated_pos3ds: calculate the 3D position of the points from the 2D positions of the point in multiple 
#     cameras by calling multi_camera_triangulation. Store the results in triangulated_pos3ds, 
#     which is a nPoint x 3 numpy array.
#    from multi_camera_triangulation import multi_camera_triangulation
    triangulated_pos3ds = multi_camera_triangulation(rvecs, tvecs, cmats, dvecs, img_points)
    print("*"*60+"\n# Step 4: calculate triangulated_pos3ds.")
    # print the triangulated 3D points
    for i in range(n_points):
        print("# Point %d: (%f, %f, %f)" % (i, triangulated_pos3ds[i, 0], triangulated_pos3ds[i, 1], triangulated_pos3ds[i, 2]))
#  5. calculate errorPos3Ds: the error between the truePos3Ds and the triangulated_pos3ds. Print it. 
    errorPos3Ds = np.linalg.norm(truePos3Ds - triangulated_pos3ds, axis=1)
    print("*"*60+"\n# Step 5: calculate errorPos3Ds.")
    # print the error of each point in this form:
    # point i: error=errorPos3Ds[i]
    for i in range(n_points):
        print("# Point %d: error=%f" % (i, errorPos3Ds[i]))
    

# gui unit test of multi_camera_triangulation
# This function creates a gui for the unit test of multi_camera_triangulation
def unit_test_gui_multi_camera_triangulation_v01():
    # create a tk window, with a given window title
    import tkinter as tk
    from tkinter import ttk
    window = tk.Tk()
    window.title("Unit Test of multi_camera_triangulation. Generated by github copilot line-by-line polished by vince")
    # the size of the window is 1000 by 700)
    window.geometry("1400x600")
    # irow initialization
    irow = 0
    # a label at left and a text entry at right for the number of cameras, with a default value. 
    label_n_cameras = ttk.Label(window, text="Number of cameras:")
    label_n_cameras.grid(row=irow, column=0)
    text_n_cameras = ttk.Entry(window)
    text_n_cameras.insert(0, "16")
    text_n_cameras.grid(row=irow, column=1)
    text_n_cameras.config(width=60)
    # a label at left and a text entry at right for the image width and height for all cameras
    irow += 1
    label_img_size = ttk.Label(window, text="Image width and height for all cameras:")
    label_img_size.grid(row=irow, column=0)
    text_img_size = ttk.Entry(window)
    text_img_size.insert(0, "800 600")
    text_img_size.grid(row=irow, column=1)
    text_img_size.config(width=60)
    # a label at left and a text entry at right for camera position x, with a default value.    
    irow += 1
    label_camera_pos_x = ttk.Label(window, text="X position of cameras:")
    label_camera_pos_x.grid(row=irow, column=0)
    text_camera_pos_x = ttk.Entry(window)
    text_camera_pos_x.insert(0, "-6 -3 0 3 6 6 6 6 6 3 0 -3 -6 -6 -6 -6")
    text_camera_pos_x.grid(row=irow, column=1)
    text_camera_pos_x.config(width=60)
    # a label at left and a text entry at right for camera position y, with a default value.
    irow += 1
    label_camera_pos_y = ttk.Label(window, text="Y position of cameras:")
    label_camera_pos_y.grid(row=irow, column=0)
    text_camera_pos_y = ttk.Entry(window)
    text_camera_pos_y.insert(0, "-6 -6 -6 -6 -6 -3 0 3 6 6 6 6 6 3 0 -3")
    text_camera_pos_y.grid(row=irow, column=1)
    text_camera_pos_y.config(width=60)
    # a label at left and a text entry at right for camera position z, with a default value.
    irow += 1
    label_camera_pos_z = ttk.Label(window, text="Z position of cameras:")
    label_camera_pos_z.grid(row=irow, column=0)
    text_camera_pos_z = ttk.Entry(window)
    text_camera_pos_z.insert(0, "3 6 6 6 3 6 6 6 3 6 6 6 3 6 6 6")
    text_camera_pos_z.grid(row=irow, column=1)
    text_camera_pos_z.config(width=60)
    # a label at left and a text entry at right for a virtual aiming target x of cameras, with a default value.
    irow += 1
    label_camera_aim_x = ttk.Label(window, text="X position of aiming target:")
    label_camera_aim_x.grid(row=irow, column=0)
    text_camera_aim_x = ttk.Entry(window)
    text_camera_aim_x.insert(0, "0 "*16)
    text_camera_aim_x.grid(row=irow, column=1)
    text_camera_aim_x.config(width=60)
    # a label at left and a text entry at right for a virtual aiming target y of cameras, with a default value.
    irow += 1
    label_camera_aim_y = ttk.Label(window, text="Y position of aiming target:")
    label_camera_aim_y.grid(row=irow, column=0)
    text_camera_aim_y = ttk.Entry(window)
    text_camera_aim_y.insert(0, "0 "*16)
    text_camera_aim_y.grid(row=irow, column=1)
    text_camera_aim_y.config(width=60)
    # a label at left and a text entry at right for a virtual aiming target z of cameras, with a default value.
    irow += 1
    label_camera_aim_z = ttk.Label(window, text="Z position of aiming target:")
    label_camera_aim_z.grid(row=irow, column=0)
    text_camera_aim_z = ttk.Entry(window)
    text_camera_aim_z.insert(0, "3 "*16)
    text_camera_aim_z.grid(row=irow, column=1)
    text_camera_aim_z.config(width=60)
    # a label at left and a text entry at right for the camera fov(x) in degrees, with a default value.
    irow += 1
    label_camera_fov_x = ttk.Label(window, text="Camera fov(x) in degrees:")
    label_camera_fov_x.grid(row=irow, column=0)
    text_camera_fov_x = ttk.Entry(window)
    text_camera_fov_x.insert(0, "100")
    text_camera_fov_x.grid(row=irow, column=1)
    text_camera_fov_x.config(width=60)
    # a label at left and a text entry at right for the camera distortion coefficients, with a default value.
    irow += 1        
    label_camera_distortion = ttk.Label(window, text="Camera distortion coefficients (k1 k2 p1 p2 k3 k4 k5 k6):")
    label_camera_distortion.grid(row=irow, column=0)
    text_camera_distortion = ttk.Entry(window)
    text_camera_distortion.insert(0, "0.01 0.01 0.01 0.01 0.01")
    text_camera_distortion.grid(row=irow, column=1)
    text_camera_distortion.config(width=60)
    # a text entry for text output (at row 1, column 2, row span 11, column span 2)
    label_general_output = ttk.Label(window, text="General Output:")
    label_general_output.grid(row=0, column=2)
    text_output = tk.Text(window, width=80, height=20)
    text_output.grid(row=1, column=2, rowspan=11, columnspan=2)

    # returns all camera parameters (a 2D array, which is n_cameras x 25) from the text entries
    # where each camera has 25 parameters: (img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z 
    # fx 0 cx 0 fy cy 0 0 1 k1 k2 p1 p2 k3 k4 k5 k6
    def get_all_cam_parms_from_text_entries():
        n_cameras = int(text_n_cameras.get())
        img_width, img_height = map(int, text_img_size.get().split())
        camera_pos_x = list(map(float, text_camera_pos_x.get().split()))
        camera_pos_y = list(map(float, text_camera_pos_y.get().split()))
        camera_pos_z = list(map(float, text_camera_pos_z.get().split()))
        camera_aim_x = list(map(float, text_camera_aim_x.get().split()))
        camera_aim_y = list(map(float, text_camera_aim_y.get().split()))
        camera_aim_z = list(map(float, text_camera_aim_z.get().split()))
        camera_fov_x = float(text_camera_fov_x.get())
        camera_distortion = list(map(float, text_camera_distortion.get().split()))
        # camera parameters in a single 2D array: all_cam_parms[i,j] is the i-th parameter of the j-th camera
        # parameters: img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z fx 0 cx 0 fy cy 0 0 1 k1 k2 p1 p2 k3 k4 k5 k6
        all_cam_parms = np.zeros((n_cameras, 25), dtype=float)
        for i in range(n_cameras): 
            icam = Camera()
            icam.imgSize = np.array((img_width, img_height)).flatten()
            icam.setCmatByImgsizeFovs(icam.imgSize, camera_fov_x)
            icam.setRvecTvecByPosAim(np.array([camera_pos_x[i], camera_pos_y[i], camera_pos_z[i]]), 
                                     np.array([camera_aim_x[i], camera_aim_y[i], camera_aim_z[i]]) )
            icam.dvec = np.array(camera_distortion).reshape(-1,1)
            all_cam_parms[i,0] = icam.imgSize[0]
            all_cam_parms[i,1] = icam.imgSize[1]
            all_cam_parms[i,2:5] = icam.rvec.flatten()
            all_cam_parms[i,5:8] = icam.tvec.flatten()
            all_cam_parms[i,8:17] = icam.cmat.flatten()
            all_cam_parms[i,17:17+icam.dvec.size] = icam.dvec.flatten()
        return all_cam_parms

    # a button to print camera parameters in a text entry in a csv format
    # header row is "/, cam_0, cam_1, ..., cam_(n-1)"
    # the first column is "img_width\n img_height\n rvec_x\n rvec_y\n rvec_z\n tvec_x\n tvec_y\n tvec_z\n fx\n fy\n cx\n cy
    # k1\n k2\n p1\n p2\n k3\n k4\n k5\n k6"
    # the second column is the camera parameters of cam_0 in the csv format, and so on.
    irow += 1
    def bt_event_print_camera_parameters():
        # get all camera parameters from the text entries
        all_cam_parms = get_all_cam_parms_from_text_entries()
        n_cameras = all_cam_parms.shape[0]
        # print the camera parameters in a csv format
        from SimpleTable import SimpleTable
        camsTable = SimpleTable()
        camsTable.table_header = ['img_width', 'img_height', 'rvec_x', 'rvec_y', 'rvec_z', 'tvec_x', 'tvec_y', 'tvec_z',
                                  'c11(fx)', 'c12', 'c13(cx)', 'c21', 'c22(fy)', 'cy', 'c31', 'c32', 'c33', 'k1', 'k2', 
                                  'p1', 'p2', 'k3', 'k4', 'k5','k6']
        camsTable.table_index = ['cam_%d' % (i+1) for i in range(n_cameras)]
        camsTable.table_data = all_cam_parms
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, camsTable.to_csv())
    button_print_cam_params = ttk.Button(window, text="Print Camera Parameters", 
                                         command=bt_event_print_camera_parameters)
    button_print_cam_params.grid(row=irow, column=0)
    # a button that plots cameras in a new matplotlib window
    irow += 1
    def bt_event_plot_cameras():
        # get all camera parameters from the text entries
        all_cam_parms = get_all_cam_parms_from_text_entries()
        # img_sizes is a list of n_cameras elements, where each element is a 2x1 numpy array
        img_sizes = [all_cam_parms[i,0:2].reshape(-1,1) for i in range(all_cam_parms.shape[0])]
        # rvecs is a list of n_cameras elements, where each element is a 3x1 numpy array
        rvecs = [all_cam_parms[i,2:5].reshape(-1,1) for i in range(all_cam_parms.shape[0])]
        # tvecs is a list of n_cameras elements, where each element is a 3x1 numpy array
        tvecs = [all_cam_parms[i,5:8].reshape(-1,1) for i in range(all_cam_parms.shape[0])]
        # cmats is a list of n_cameras elements, where each element is a 3x3 numpy array
        cmats = [all_cam_parms[i,8:17].reshape(3,3) for i in range(all_cam_parms.shape[0])]
        # dvecs is a list of n_cameras elements, where each element is a 1x5 or 1x8 numpy array
        dvecs = [all_cam_parms[i,17:].reshape(-1,1) for i in range(all_cam_parms.shape[0])]
        # plot the cameras
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from plotCameras3d import plot_cameras
        plot_cameras(rvecs, tvecs, cmats, dvecs, img_sizes, h_cone=1, axes=None, 
                 xlim=None, ylim=None, zlim=None, 
                 equal_aspect_ratio=True)
        plt.show()
    button_plot_cameras = ttk.Button(window, text="Plot Cameras", command=bt_event_plot_cameras)
    button_plot_cameras.grid(row=irow, column=0)
    # About points
    # a label with the text "About Points"
    irow += 1
    label_about_points = ttk.Label(window, text="About Points")
    label_about_points.grid(row=irow, column=0)
    # a label at left and a text entry at right for the center of the cube x y z, with a default value.
    irow += 1
    label_center = ttk.Label(window, text="Center of the points (in shape of a cube):")
    label_center.grid(row=irow, column=0)
    text_center = ttk.Entry(window)
    text_center.insert(0, "0 0 3 ")
    text_center.grid(row=irow, column=1)
    text_center.config(width=60)
    # a label at left and a text entry at right for the edge (length) of this cube
    irow += 1
    label_edge = ttk.Label(window, text="Edge of the cube:")
    label_edge.grid(row=irow, column=0)
    text_edge = ttk.Entry(window)
    text_edge.insert(0, "6.0")
    text_edge.grid(row=irow, column=1)
    text_edge.config(width=60)
    # a label at left and a text entry at right for number of points per edge of the cube
    irow += 1
    label_npoints = ttk.Label(window, text="Number of points per edge of the cube:")
    label_npoints.grid(row=irow, column=0)
    text_npoints = ttk.Entry(window)
    text_npoints.insert(0, "8")
    text_npoints.grid(row=irow, column=1)
    text_npoints.config(width=60)
    # a button that prints the 3D points to the text output in a csv format
    # the header row is ",x, y, z"
    # the index column is "Point 1" ... to "Point N" where N is the number of points
    # the data is the 3D points in the shape of a cube
    irow += 1
    def bt_event_print_points():
        # get the center of the points
        center = np.array(list(map(float, text_center.get().split())))
        # get the edge of the cube
        edge = float(text_edge.get())
        # get the number of points per edge of the cube
        npoints = int(text_npoints.get())
        # calculate the 3D points in the shape of a cube
        from create_synthetic_3d_points_cube import create_synthetic_3d_points_cube
        allpoints = create_synthetic_3d_points_cube(center=center, edge=edge, n_points_per_edge=npoints)
        # print all points in a csv format
        from SimpleTable import SimpleTable
        pointsTable = SimpleTable()
        pointsTable.table_header = ['x', 'y', 'z']
        pointsTable.table_index = ['Point %d' % (i+1) for i in range(allpoints.shape[0])]
        pointsTable.table_data = allpoints
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, pointsTable.to_csv())
    button_print_points = ttk.Button(window, text="Print points", 
                                         command=bt_event_print_points)
    button_print_points.grid(row=irow, column=0)
    # a button that plots the points in a new matplotlib window
    irow += 1
    def bt_event_plot_points():
        # get the center of the points
        center = np.array(list(map(float, text_center.get().split())))
        # get the edge of the cube
        edge = float(text_edge.get())
        # get the number of points per edge of the cube
        npoints = int(text_npoints.get())
        # calculate the 3D points in the shape of a cube
        from create_synthetic_3d_points_cube import create_synthetic_3d_points_cube
        pos3Ds = create_synthetic_3d_points_cube(center=center, edge=edge, n_points_per_edge=npoints)
        # plot the points
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos3Ds[:,0], pos3Ds[:,1], pos3Ds[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Points in the shape of a cube')
        plt.show()
    button_plot_points = ttk.Button(window, text="Plot Points", command=bt_event_plot_points)
    button_plot_points.grid(row=irow, column=0)
    # a button titled with "Print and show synthetic image"
    # This button creates an image that is taken by the camera with the given index
    irow += 1
    def bt_event_synthetic_image_taken_by_camera():
        # get the camera index
        icam = int(text_camera_index.get())
        # get all camera parameters from the text entries
        all_cam_parms = get_all_cam_parms_from_text_entries()
        img_size = all_cam_parms[icam-1,0:2].astype(int)
        rvec = all_cam_parms[icam-1,2:5].reshape(-1,1)
        tvec = all_cam_parms[icam-1,5:8].reshape(-1,1)
        cmat = all_cam_parms[icam-1,8:17].reshape(3,3)
        dvec = all_cam_parms[icam-1,17:].reshape(-1,1)
        # get all points
        center = np.array(list(map(float, text_center.get().split())))
        edge = float(text_edge.get())
        npoints_per_edge = int(text_npoints.get())
        from create_synthetic_3d_points_cube import create_synthetic_3d_points_cube
        allpoints = create_synthetic_3d_points_cube(center=center, edge=edge, n_points_per_edge=npoints_per_edge)
        # create a white blank image with the given size (img_size) in OpenCV format
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        # project all points to the image. img_points is a n_points x 2 numpy array
        img_points = cv2.projectPoints(allpoints, rvec, tvec, cmat, dvec)[0].reshape(-1,2)
        # print img_points to the text output
        from SimpleTable import SimpleTable
        prjTable = SimpleTable()
        prjTable.table_header = ['xi_projected', 'yi_projected']
        prjTable.table_index = ['point_%d' % (i+1) for i in range(img_points.shape[0])]
        prjTable.table_data = img_points
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, prjTable.to_csv())
        # draw all points in the image. Each point is a blue circle with radius 5
        radius = 5
        for i in range(allpoints.shape[0]):
            cv2.circle(img, tuple(img_points[i].astype(int).flatten()), radius, (255, 0, 0), -1)
        # show the image
        from imshow2 import imshow2
        imshow2(winname="Synthetic Image Taken by Camera %d" % icam, img=img, winmax=(1600,700))
    button_synthetic_image_taken_by_camera = ttk.Button(window, text="Print and show synthetic image of camera (1-based index):", 
                                         command=bt_event_synthetic_image_taken_by_camera)
    button_synthetic_image_taken_by_camera.grid(row=irow, column=0)
    # a text entry at the right for the camera index with a default value 1, 
    text_camera_index = ttk.Entry(window)
    text_camera_index.insert(0, "1")
    text_camera_index.grid(row=irow, column=1)
    text_camera_index.config(width=10)
    # a button titled with "multi_camera_triangulation"
    # This button calculates the 3D points from the 2D points in multiple cameras
    # by calling multi_camera_triangulation. The 3D points are printed to the text output.
    # a text entry at the right for the noise level (unit: pixel) with a default value 1.0  
    irow += 1
    text_image_point_noise = ttk.Entry(window)
    text_image_point_noise.insert(0, "1.0")
    text_image_point_noise.grid(row=irow, column=1)
    text_image_point_noise.config(width=10)
    # a button 
    def bt_event_multi_camera_triangulation():
        rvecs=[]; tvecs=[]; cmats=[]; dvecs=[]; img_pointss=[]; img_sizes=[]
        
        # get all camera parameters from the text entries
        all_cam_parms = get_all_cam_parms_from_text_entries()
        for icam in range(0, int(text_n_cameras.get())):
            img_size = all_cam_parms[icam-1,0:2].astype(int)
            rvec = all_cam_parms[icam-1,2:5].reshape(-1,1)
            tvec = all_cam_parms[icam-1,5:8].reshape(-1,1)
            cmat = all_cam_parms[icam-1,8:17].reshape(3,3)
            dvec = all_cam_parms[icam-1,17:].reshape(-1,1)
            img_sizes.append(img_size)
            rvecs.append(rvec)
            tvecs.append(tvec)
            cmats.append(cmat)
            dvecs.append(dvec)
        # get all points
        center = np.array(list(map(float, text_center.get().split())))
        edge = float(text_edge.get())
        npoints_per_edge = int(text_npoints.get())
        from create_synthetic_3d_points_cube import create_synthetic_3d_points_cube
        allpoints = create_synthetic_3d_points_cube(center=center, edge=edge, n_points_per_edge=npoints_per_edge)
        # project all points to the image. img_points is a n_points x 2 numpy array
        for icam in range(0, int(text_n_cameras.get())):
            img_points = cv2.projectPoints(allpoints, rvecs[icam], tvecs[icam], cmats[icam], dvecs[icam])[0].reshape(-1,2)
            # add noise
            noise = float(text_image_point_noise.get())
            img_points = img_points + np.random.randn(img_points.shape[0], img_points.shape[1]) * noise
            img_pointss.append(img_points)
        # calculate the 3D points from the 2D points in multiple cameras
        triangulated_points, projected_pointss = multi_camera_triangulation(rvecs, tvecs, cmats, dvecs, img_pointss)
        # convert img_pointss (a list) to a 2D numpy array for printing. 
        # img_points_table[ipnt, icam*2:icam*2+2] is the image point of the ipnt-th point in the icam-th camera
        # That is, img_points_table[0] contains xi_cam1, yi_cam_1, xi_cam_2, yi_cam_2, ..., xi_cam_N, yi_cam_N 
        img_points_for_table = np.concatenate(img_pointss, axis=1)
        # convert projected_pointss (a list) to a 2D numpy array for printing. 
        # projected_pointss_table[ipnt, icam*2:icam*2+2] is the projected point of the ipnt-th point in the icam-th camera
        # That is, projected_pointss_table[0] contains xi_cam1, yi_cam_1, xi_cam_2, yi_cam_2, ..., xi_cam_N, yi_cam_N 
        projected_pointss_for_table = np.concatenate(projected_pointss, axis=1)
        error_for_table = projected_pointss_for_table - img_points_for_table
        # triangulated_points_table combines triangulated_points, img_points_table, projected_pointss_table, and error_table
        triangulated_points_for_table = np.concatenate((triangulated_points, img_points_for_table, projected_pointss_for_table, error_for_table), axis=1)
        # print the triangulated 3D points to the text output, also image points (xi), projected points (xp), and projected error (err)
        from SimpleTable import SimpleTable
        triangulated_table = SimpleTable()
        triangulated_table.table_header = ['xw', 'yw', 'zw']
        for icam in range(0, int(text_n_cameras.get())):
            triangulated_table.table_header.append('xi_cam_%d' % (icam+1))
            triangulated_table.table_header.append('yi_cam_%d' % (icam+1))
        for icam in range(0, int(text_n_cameras.get())):
            triangulated_table.table_header.append('xp_cam_%d' % (icam+1))
            triangulated_table.table_header.append('yp_cam_%d' % (icam+1))
        for icam in range(0, int(text_n_cameras.get())):
            triangulated_table.table_header.append('xerr_cam_%d' % (icam+1))
            triangulated_table.table_header.append('yerr_cam_%d' % (icam+1))
        triangulated_table.table_index = ['point_%d' % (i+1) for i in range(triangulated_points.shape[0])]
        triangulated_table.table_data = triangulated_points_for_table
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, triangulated_table.to_csv())
        # return
    button_multi_camera_triangulation = ttk.Button(window, text="multi_camera_triangulation with projection noise (in pixels):", 
                                         command=bt_event_multi_camera_triangulation)
    button_multi_camera_triangulation.grid(row=irow, column=0)
    # a button that prints 
        




    
    
    # run the gui program
    window.mainloop()

# main program
if __name__ == '__main__':
    unit_test_gui_multi_camera_triangulation_v01()
