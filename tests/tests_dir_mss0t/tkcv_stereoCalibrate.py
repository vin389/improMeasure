import numpy as np
import cv2
import tkinter as tk
from inputdlg3 import inputdlg3 
from chessboard_object_points import chessboard_object_points

def tkcv_stereoCalibrate_from_files(initvalues=None):
    # popup a tk input dialog (inputdlg3) to get parameters, including:
    #   pattern size (numbers of corners in the pattern) and square size (size of each square in the pattern)
    #       This is to generate the object points for calibration
    #       for example, 14 9 12.1 12.1 (meaning 14 corners in x direction, 9 corners in y direction, and each square is 12.1 mm)
    #       will generate object points like:
    #       [[0, 0, 0], [12.1, 0, 0], [24.2, 0, 0], ..., [12.1*13, 12.1*8, 0]] * <num_photos> and reshape to (<num_photos>, -1, 3)
    #   file of image points of camera 1
    #       This is a (csv) file with 2D points, each point in a separate line (with 2 columns, x and y coordinates)
    #   file image points of camera 2
    #       This is a (csv) file with 2D points, each point in a separate line (with 2 columns, x and y coordinates)
    #   file of camera parameters of camera 1 
    #       parameters are saved in the format in single column
    #       (img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z 
    #        cmat_00 cmat_01 cmat_02 cmat_10 cmat_11 cmat_12 cmat_20 cmat_21 cmat_22 
    #        k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y) 
    #   file of camera parameters of camera 2 
    #       parameters are saved in the format in single column
    #       (img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z 
    #        cmat_00 cmat_01 cmat_02 cmat_10 cmat_11 cmat_12 cmat_20 cmat_21 cmat_22 
    #        k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y) 
    #   flags 
    #       flags are input by user through listbox 
    #   term criteria (count eps)
    #       term criteria are input by user through entry boxes 
    #       non-positive value means it is not used.
    #       for example, TermCriteria(cv2.TermCriteria_COUNT+cv2.TermCriteria_EPS, 30, 0.001) (30 iterations and epsilon of 0.001)
    #       0  0.001 means TermCriteria(cv2.TermCriteria_EPS, 0, 0.001) 
    #       30 0 means TermCriteria(cv2.TermCriteria_COUNT, 30, 0.000) 
    #       default value is 30 0.001 

    strFlags = ['CALIB_USE_INTRINSIC_GUESS', 
                'CALIB_FIX_ASPECT_RATIO',
                'CALIB_FIX_PRINCIPAL_POINT', 
                'CALIB_ZERO_TANGENT_DIST',
                'CALIB_FIX_FOCAL_LENGTH', 
                'CALIB_FIX_K1',
                'CALIB_FIX_K2', 
                'CALIB_FIX_K3', 
                'CALIB_FIX_K4',
                'CALIB_FIX_K5', 
                'CALIB_FIX_K6', 
                'CALIB_RATIONAL_MODEL',
                'CALIB_THIN_PRISM_MODEL', 'CALIB_FIX_S1_S2_S3_S4',
                'CALIB_TILTED_MODEL', 'CALIB_FIX_TAUX_TAUY', 'CALIB_USE_QR',
                'CALIB_FIX_TANGENT_DIST', 
                'CALIB_FIX_INTRINSIC',
                'CALIB_SAME_FOCAL_LENGTH', 
                'CALIB_ZERO_DISPARITY',
                'CALIB_USE_LU', 
                'CALIB_USE_EXTRINSIC_GUESS' ]


    prompts = ["Pattern size", 
               "Image points 1", "Image points 2", 
               "Parameters of camera 1", "Parameters of camera 2",
               "Flags", "Term criteria",
               "Camera 1 output file ", "Camera 2 output file"
               ]
    datatypes = ["array 1 -1 ", 
                 "file", "file",
                 "file", "file",
                 "listbox_h12 " + " ".join(strFlags), "array 1 2", 
                 "filew", "filew"] 
    if initvalues is None:
        # default values for the input dialog
        # pattern size: 12 corners in x, 9 corners in y, square size 12.1 mm
        initvalues = ["12 9 12.1 12.1", 
                    "c:/image_points_corners_cam1.csv", "c:/image_points_corners_cam2.csv",
                    "c:/camera_parameters_1_col_format.txt", "c:/camera_parameters_2_col_format.txt",
                    [0] * len(strFlags), "30 0.001"
                    "c:/camera_parameters_cam1_local.csv", "c:/camera_parameters_cam2_local.csv"
                    ]
    tooltips = ["Pattern size (corners in x, corners in y, square size in mm)",
                "Image points of camera 1 (csv file with 2 columns. If they are from 10 pictures of a 12x9 chessboard, there will be 10*12*9= 1080 points)",
                "Image points of camera 2 (csv file with 2 columns. If they are from 10 pictures of a 12x9 chessboard, there will be 10*12*9= 1080 points)",
                "Camera 1 parameters (file with single column format: img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z cmat_00 cmat_01 cmat_02 cmat_10 cmat_11 cmat_12 cmat_20 cmat_21 cmat_22 k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y)",
                "Camera 2 parameters (file with single column format: img_width img_height rvec_x rvec_y rvec_z tvec_x tvec_y tvec_z cmat_00 cmat_01 cmat_02 cmat_10 cmat_11 cmat_12 cmat_20 cmat_21 cmat_22 k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y)",
                "Flags (cv2.StereoCalibrateFlags)",
                "Term criteria (count eps)",
                "Camera 1 output file (to save the camera parameters after calibration, local, extrinsic parameters to be zeros)",
                "Camera 2 output file (to save the camera parameters after calibration, local, extrinsic parameters is relative to camera 1)"
                ]
    result = inputdlg3(prompts=prompts, datatypes=datatypes, 
                       initvalues=initvalues, tooltips=tooltips, 
                       title="Stereo Calibration (cv2.stereoCalibrate)")
    # analyze the inputdlg3 result from user
    # if user cancelled the dialog, result will be None
    if result is None:
        return None
    #
    # result[0]: Pattern size (numbers of corners nCornersX, nCornersY and square size dx, dy) 
    # set objPoints to 
    # --> object_points shape: ((nCbPhotos, nCornersX * nCornersY, 3)), astype(np.float32)
    try:
        pattern_size = result[0].flatten()
        nCornersX, nCornersY, dxCorners, dyCorners = (int(pattern_size[0]), int(pattern_size[1]), float(pattern_size[2]), float(pattern_size[3]))
        corners_object_points_one_picture = chessboard_object_points(nCornersX, nCornersY, dxCorners, dyCorners)
        # corners_object_points = np.tile(corners_object_points_one_picture, (num_cb_photos, 1, 1))
    except:
        print("Invalid pattern size input. Please enter valid numbers for corners and square size.")
        return None
    #
    # result[1]: Image points of camera 1 (reshaped to <num_cb_photos>, -1, 2)
    try:
        # result[1] is a csv file. Load the file and convert it to a (num_cb_photos, nCornersX * nCornersY, 2) array
        image_points_1 = np.loadtxt(result[1], delimiter=',').astype(np.float32).reshape(-1,2)
        num_cb_photos = image_points_1.shape[0] // (nCornersX * nCornersY)
        if num_cb_photos * (nCornersX * nCornersY) != image_points_1.shape[0]:
            print("Invalid image points 1 input. The number of points does not match the expected number of corners.")
            return None
        # reshape image_points_1 
        image_points_1 = image_points_1.reshape((num_cb_photos, nCornersX * nCornersY, 2))
    except:
        print("Invalid image points 1 input. Please enter valid 2D points.")
        return None
    # result[2]: Image points of camera 2 (reshaped to <num_cb_photos>, -1, 2)
    try:
        image_points_2 = np.loadtxt(result[2], delimiter=',').astype(np.float32).reshape(-1,2)
        num_cb_photos_2 = image_points_2.shape[0] // (nCornersX * nCornersY)
        if num_cb_photos_2 != num_cb_photos:
            print("Invalid image points 2 input. The number of points does not match the number of points in camera 1.")
            return None
        # reshape image_points_2
        image_points_2 = image_points_2.reshape((num_cb_photos, nCornersX * nCornersY, 2))
    except:
        print("Invalid image points 2 input. Please enter valid 2D points.")
        return None
    # build the object points for all images
    corners_object_points = np.tile(corners_object_points_one_picture, (num_cb_photos, 1, 1)).astype(np.float32)
    # Load camera 1 parameters from file result[3] 
    try:
        with open(result[3], 'r') as f:
            # load by numpy 
            camera_1_params = np.genfromtxt(result[3], delimiter=r'[ ,;\t]+', comments='#')
            img_size_1 = [int(camera_1_params[0]), int(camera_1_params[1])]
            cmat1 = camera_1_params[8:17].reshape((3, 3))
            dvec1 = camera_1_params[17:].reshape((1, -1))
    except:
        print("Invalid camera parameters for camera 1. Please enter valid file that contains img_w img_h rvec (3 values) tvec (3 values) cmat (9 values) dvec (1x num_coefficients: 4, 5, or 8).")
        return None
    # Load camera 2 parameters from file result[4] 
    try:
        with open(result[4], 'r') as f:
            # load by numpy 
            camera_2_params = np.genfromtxt(result[4], delimiter=r'[ ,;\t]+', comments='#')
            img_size_2 = [int(camera_2_params[0]), int(camera_2_params[1])]
            cmat2 = camera_2_params[8:17].reshape((3, 3))
            dvec2 = camera_2_params[17:].reshape((1, -1))
    except:
        print("Invalid camera parameters for camera 2. Please enter valid file that contains img_w img_h rvec (3 values) tvec (3 values) cmat (9 values) dvec (1x num_coefficients: 4, 5, or 8).")
        return None
    # result[5]: Flags (cv2.StereoCalibrateFlags)
    try:
        flags = 0
        #  result[8] lists the flags that are selected by the user
        #  for example, if result[8] = [0, 7, 8], it means the user selected CALIB_USE_INTRINSIC_GUESS, CALIB_FIX_K1, and CALIB_FIX_K2
        #  then we set CALIB_USE_INTRINSIC_GUESS = 1, CALIB_FI
        for idx_flag in result[5]:
            flag = strFlags[idx_flag]
            flags |= getattr(cv2, flag)
    except:
        print("Invalid flags input. Please enter valid flags from the list.")
        return None
    # result[9]: Term criteria (count eps)
    try:
        term_criteria = np.array(result[6]).astype(np.float32).flatten()
        if term_criteria.size != 2:
            print("Invalid term criteria input. Please enter valid count and epsilon.")
            return None
        term_criteria_count = int(term_criteria[0])
        term_criteria_eps = float(term_criteria[1])
        if term_criteria_count > 0 and term_criteria_eps > 0.0:
            term_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, term_criteria_count, term_criteria_eps)
        elif term_criteria_count > 0:
            term_criteria = (cv2.TERM_CRITERIA_COUNT, term_criteria_count, 0.0)
        elif term_criteria_eps > 0.0:
            term_criteria = (cv2.TERM_CRITERIA_EPS, 0, term_criteria_eps)
        else:
            term_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)
    except:
        print("Invalid term criteria input. Please enter valid count and epsilon.")
        return None
    # result[10] and result[11]: Camera 1 output file and Camera 2 output file
    try:
        camera_1_output_file = result[7]
        camera_2_output_file = result[8]
    except:
        print("Invalid camera output file paths.")
        return None 
    # Now we have all the inputs validated and ready for stereo calibration
    # Call cv2.stereoCalibrate
    try:
        # if the cv2.CALIB_FIX_INTRINSIC_GUESS flag is not set, 
        # calibration camera 1 
        if not (flags & cv2.CALIB_FIX_INTRINSIC):
            calib_result_1 = cv2.calibrateCamera(
                corners_object_points, 
                image_points_1, 
                img_size_1,
                cmat1, dvec1,
                None, None, 
                flags,
                term_criteria
            )
            retval_1, cmat1, dvec1, rvecs_1, tvecs_1 = calib_result_1
            if retval_1 == 0:
                print("Calibration of camera 1 failed.")
                return None
            # calibration step 2: calibrateCamera of camera 2
            calib_result_2 = cv2.calibrateCamera(
                corners_object_points,
                image_points_2,
                img_size_2, 
                cmat2, dvec2,
                None, None, 
                flags,
                term_criteria
            )
            retval_2, cmat2, dvec2, rvecs_2, tvecs_2 = calib_result_2
            if retval_2 == 0:
                print("Calibration of camera 2 failed.")
                return None 
        # end if cv2.CALIB_FIX_INTRINSIC_GUESS flag is not set

        # calibration step 3
        calib_result = cv2.stereoCalibrate(
            corners_object_points, 
            image_points_1,
            image_points_2,
            cmat1, dvec1,
            cmat2, dvec2,
            img_size_1,
            flags,
            term_criteria
        )
    except:
        print("Stereo calibration failed.")
        return None
    # If calibration was successful, return the results.
    # The results include 
    # retval, cmat1, dvec1, cmat2, dvec2, R, T, E, F 
    retval, cmat1_new, dvec1_new, cmat2_new, dvec2_new, R, T, E, F = calib_result
    # if intrinsic parameters are not fixed, we need to update cmat1 and dvec1
    if not (flags & cv2.CALIB_FIX_INTRINSIC):
        cmat1 = cmat1_new
        dvec1 = dvec1_new
        cmat2 = cmat2_new
        dvec2 = dvec2_new
    # write the results to the output files
    # The format is one column format: img_width img_height rvec (3 values) tvec (3 values) cmat (9 values) dvec (1x num_coefficients: 4, 5, or 8)
    # For camera 1, the rvec and tvec are zeros
    # For camera 2, the rvec and tvec are relative to camera 1
    if retval == 0:
        print("Stereo calibration failed.")
        return None
    else:
        print("Stereo calibration successful.")
        with open(camera_1_output_file, 'w') as f1:
            # write image width and height
            f1.write("# image width and height\n")
            f1.write(" %d\n %d\n" % (img_size_1[0], img_size_1[1]))
            # write rvec and tvec for camera 1, which are zeros
            f1.write("# rvec and tvec for camera 1 (zeros because it is relative to camera 1 itself)\n")
            f1.write(" 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n 0.0\n")
            # write camera matrix (9 values)
            f1.write("# intrinsic parameters 3x3 camera matrix\n")
            f1.write("# fx 0 cx 0 fy cy 0 0 1\n")
            for i in range(9):
                f1.write(" %.6f\n" % cmat1.flatten()[i])
            # write distortion coefficients (1x num_coefficients)
            f1.write("# distortion coefficients: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 taux tauy\n")
            for i in range(dvec1.size):
                f1.write(" %.6f\n" % dvec1.flatten()[i])
            # additional information (camera location relative to camera 1)
            this_rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            this_tvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            this_r44 = np.eye(4, dtype=np.float32)
            this_r44[:3, :3] = cv2.Rodrigues(this_rvec)[0]  # Convert rotation vector to rotation matrix
            this_r44[0:3, 3] = this_tvec
            this_r44_inv = np.linalg.inv(this_r44)
            f1.write("# camera location relative to camera 1 (4x4 matrix)\n")
            f1.write("# %f %f %f \n" % (this_r44_inv[0, 3], this_r44_inv[1, 3], this_r44_inv[2, 3]))
            # print message
            print("Camera 1 parameters saved to:", camera_1_output_file)  
        with open(camera_2_output_file, 'w') as f1:
            # write image width and height
            f1.write("# image width and height\n")
            f1.write(" %d\n %d\n" % (img_size_2[0], img_size_2[1]))
            # write rvec and tvec for camera 2
            f1.write("# rvec and tvec for camera 2 (relative to camera 1)\n")
            rvec = cv2.Rodrigues(R)[0]  # Convert rotation matrix to rotation vector
            for i in range(3):
                f1.write(" %.6f\n" % rvec.flatten()[i])
            for i in range(3):
                f1.write(" %.6f\n" % T.flatten()[i])
            # write camera matrix (9 values)
            f1.write("# intrinsic parameters 3x3 camera matrix\n")
            f1.write("# fx 0 cx 0 fy cy 0 0 1\n")
            for i in range(9):
                f1.write(" %.6f\n" % cmat2.flatten()[i])
            # write distortion coefficients (1x num_coefficients)
            f1.write("# distortion coefficients: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 taux tauy\n")
            for i in range(dvec2.size):
                f1.write(" %.6f\n" % dvec2.flatten()[i])
            # additional information (camera location relative to camera 1)
            this_rvec = rvec.flatten()
            this_tvec = T.flatten()
            this_r44 = np.eye(4, dtype=np.float32)
            this_r44[:3, :3] = cv2.Rodrigues(this_rvec)[0]  # Convert rotation vector to rotation matrix
            this_r44[0:3, 3] = this_tvec
            this_r44_inv = np.linalg.inv(this_r44)
            f1.write("# camera location relative to camera 1 (4x4 matrix)\n")
            f1.write("# %f %f %f \n" % (this_r44_inv[0, 3], this_r44_inv[1, 3], this_r44_inv[2, 3]))
            # print message
            print("Camera 2 parameters saved to:", camera_2_output_file)
        # end of with open camera_2_output_file
    return calib_result

def brb1_test2():
    initvalues = ["7 12 7 12 21.310000 21.350000", 
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam1_northGusset_calib/brb1_cam1_found_corners_13_photos_7x12_pattern.csv",
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam2_northGusset_calib/brb1_cam2_found_corners_13_photos_7x12_pattern.csv",
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam1_northGusset_calib/tkCalib2_chessboard_intrinsic_k1_k2_p1_p2.txt",
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam2_northGusset_calib/tkCalib2_chessboard_intrinsic_k1_k2_p1_p2.txt",
                  [0,  # CALIB_USE_INTRINSIC_GUESS
                   1,  # CALIB_FIX_ASPECT_RATIO
                   2,  # CALIB_FIX_PRINCIPAL_POINT
                   7,  # CALIB_FIX_K3
                   10, # CALIB_FIX_K6
                   11, # CALIB_RATIONAL_MODEL
                   18, # CALIB_FIX_INTRINSIC
                   ], 
                  "30 0.001",
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam1_northGusset_calib/brb1_cam1_local_parameters.csv",
                  "D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam2_northGusset_calib/brb1_cam2_local_parameters.csv"
    ]


    results = tkcv_stereoCalibrate_from_files(initvalues)
    if results is None:
        print("User cancelled the dialog.")
    else:
        # Print the results in a readable format
        retval, cmat1, dvec1, cmat2, dvec2, R, T, E, F = results
        print("Results from stereoCalibrate dialog:")
        for res in results:
            print(res)


# main program
if __name__ == "__main__":
    brb1_test2()