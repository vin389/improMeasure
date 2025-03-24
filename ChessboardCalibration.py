import numpy as np
import cv2
import glob
import os

from chessboard_object_points import chessboard_object_points

# The class ChessboardCalibration is a camera calibration class that calibrates
# a camera from photos taken by the camera. It is designed to integrate with 
# the following opencv functions:
#   cv2.findChessboardCorners(), 
#   cv2.calibrateCamera(),
#   cv2.projectPoints(),
#   cv2.undistortImage()
# The major features include the following:
#   1. Chessboard corner detection: The class can detect the corners of a
#      chessboard pattern from photos taken by the camera.
#   2. Camera calibration: The class can calibrate the camera from photos
#      taken by the camera.
#   3. Camera calibration quality evaluation: The class can evaluate the
#      quality of camera calibration (by reprojection error).
#   4. Camera calibration result visualization: The class can visualize the
#      camera calibration result (by undistorting the photos).

class ChessboardCalibration:
    def __init__(self):
        self.work_dir = ''
        self.photo_files = [] # list of photos which path is related to self.work_dir
        self.num_corners = (7, 7) # number of corners of the chessboard pattern
        self.square_size = 1.0 # size of the squares of the chessboard pattern (in phyical length unit like m or mm)
        self.image_points = [] # list of image points of the chessboard pattern. 
            # Length of the list is the same as the number of photos.
        self.object_points = [] # list of object points of the chessboard pattern
        self.image_size = [] # image size of the photos
        self.cmat = None
        self.dvec = None
        self.rvecs = []   # list of rotation vectors. The length of the list is 
                          # the same as the number of photos. 
        self.tvecs = []   # list of translation vectors. The length of the list is
                          # the same as the number of photos.
        self.projected_image_points = [] # list of projected image points of the
            # chessboard pattern. The same size and dimension as self.image_points.
        self.undistorted_photos = [] # list of undistorted photos. The same size
    
    # Convert the object of ChessboardCalibration to a string so that this object
    # can be printed.
    def __str__(self):
        the_str = ''
        the_str += 'work_dir: ' + self.work_dir + '\n'
        the_str += 'number of photos: ' + str(len(self.photo_files)) + '\n'
        the_str += 'photo_files: ' + str(self.photo_files) + '\n'
        the_str += 'number of corners: ' + str(self.num_corners) + '\n'
        the_str += 'square_size: ' + str(self.square_size) + '\n'
        the_str += 'number of image points: ' + str(len(self.image_points)) + '\n'
        the_str += 'image_points: ' + str(self.image_points) + '\n'
        the_str += 'object_points: ' + str(self.object_points) + '\n'
        the_str += 'image_size: ' + str(self.image_size) + '\n'
        the_str += 'cmat: ' + str(self.cmat) + '\n'
        the_str += 'dvec: ' + str(self.dvec) + '\n'
        the_str += 'number of rvecs: ' + str(len(self.rvecs)) + '\n'
        the_str += 'rvecs: ' + str(self.rvecs) + '\n'
        the_str += 'number of tvecs: ' + str(len(self.tvecs)) + '\n'
        the_str += 'tvecs: ' + str(self.tvecs) + '\n'
        the_str += 'number of projected image points: ' + str(len(self.projected_image_points)) + '\n'
        the_str += 'projected_image_points: ' + str(self.projected_image_points) + '\n'
        the_str += 'number of undistorted photos: ' + str(len(self.undistorted_photos)) + '\n'
        the_str += 'undistorted_photos: ' + str(self.undistorted_photos) + '\n'
        return the_str
        
    # Refresh (update) self.photo_files from the working directory (self.work_dir).
    # That is, to find all image files in the working directory and store them in
    # self.photo_files. The supported files include .jpg, .jpeg, .png, .bmp, .tif, 
    # and .tiff. 
    def refresh_photo_files(self):
        self.photo_files = []
        # find all files under self.work_dir with the supported file extensions
        # and store them in self.photo_files
        extensions = ['*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in extensions:
            self.photo_files.extend(glob.glob(os.path.join(self.work_dir, '**', ext), recursive=False))
        # make all files in self.photo_files to be related path with respect to self.work_dir
        self.photo_files = [os.path.relpath(photo_file, self.work_dir) for photo_file in self.photo_files]
    
    # find the corners of the chessboard pattern from the photo file, and store
    # them into self.image_points.
    # photo_indices: list of indices of the photos in self.photo_files. If it is
    # empty, all photos in self.photo_files will be processed.
    # The indices are zero based. 
    # For example: 
    #     an_ChessboardCalibration.find_corners([0, 2])
    # will find the corners of the first and third photos in self.photo_files.
    def find_corners(self, photo_indices=[], do_subpixel=False):
        # check photo_indices. If it is a single number, like 3 or 3.0, make it a list
        if isinstance(photo_indices, (int, float)):
            photo_indices = [int(photo_indices)]
        # check photo_indices. If it is empty, make it to be all photos.
        if len(photo_indices) == 0:
            photo_indices = range(len(self.photo_files))
        # clear self.image_points
        self.image_points = []
        # find corners for each photo in photo_indices
        for photo_index in photo_indices:
            photo_file = self.photo_files[photo_index]
            abs_photo_file = os.path.join(self.work_dir, photo_file)
            img = cv2.imread(abs_photo_file)
            # set self.image_size
            if (self.image_size == []):
                self.image_size = img.shape[1::-1]
            else:
                # if it has been set, check if the image size is the same as the current image
                if self.image_size != img.shape[1::-1]:
                    print('Error: The image size of %s (%d,%d) is different from the previous ones (%d,%d).' 
                          % (photo_file, img.shape[1], img.shape[0], self.image_size[0], self.image_size[1]))
                    return
            # convert the image to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.num_corners, None)
            if ret:
                self.image_points.append(corners)
                print('# There are %d corners found in %s' % (corners.shape[0], photo_file))
                # if corners found, do subpixel corner detection
                if do_subpixel:
                    print("# Doing subpixel corner detection.")
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    cv2.cornerSubPix(gray, corners, self.num_corners, (3, 3), criteria)
            else:
                self.image_points.append(None)
                print('# Corners not found in %s' % photo_file)

    # plot the corners of the chessboard pattern on the photo file
    def imshow_corners(self, photo_index=0):
        from imshow2 import imshow2
        abs_photo_file = os.path.join(self.work_dir, self.photo_files[photo_index])
        img = cv2.imread(abs_photo_file)
        if self.image_points[photo_index] is not None:
            img = cv2.drawChessboardCorners(img, self.num_corners, self.image_points[photo_index], True)
        base_name = os.path.basename(abs_photo_file)
        imshow2('Corners of %s' % base_name , img, winmax=(1600,700))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # define object points (object_points) from the number of corners 
    # (self.num_corners[0], self.num_corners[1]), and the size of the squares 
    # self.square_size[0] and [1] of the chessboard pattern.
    def set_object_points(self):
        # define object_points
        n_success_photos = len(self.image_points)
        self.object_points = []
        for i in range(n_success_photos):
            object_points = chessboard_object_points(self.num_corners[0], self.num_corners[1], self.square_size, self.square_size, face='xy')
            self.object_points.append(object_points)

    # calibrate the camera from the image points of detected corners
    # The camera calibration result will be stored in self.cmat, self.dvec,
    # self.rvecs, and self.tvecs. 
    def calibrate_camera(self):
        # define object points (object_points)
        self.set_object_points()
        # check if the number of image points and object points are the same
        if len(self.image_points) != len(self.object_points):
            print('Error: The number of sets of image points and object points are different.')
            print('       image points (%d). object points (%d).' % (len(self.image_points), len(self.object_points)))
            return
        # check if the image size is available
        if len(self.image_size) == 0:
            print('Error: The image size is not available.')
            return
        # calibrate the camera
        self.cmat = np.eye(3)
        self.dvec = np.zeros((1, 5))
        self.rvecs = []
        self.tvecs = []
        object_points = np.array(self.object_points).reshape(
            -1, self.num_corners[0] * self.num_corners[1], 3).astype(np.float32)
        image_points = np.array(self.image_points).reshape(
            -1, self.num_corners[0] * self.num_corners[1], 2).astype(np.float32)
        ret, self.cmat, self.dvec, self.rvecs, self.tvecs = cv2.calibrateCamera(
            object_points, image_points, self.image_size, 
            self.cmat, self.dvec, self.rvecs, self.tvecs)
        print('Camera calibration done.')
        print('ret: ', ret)
        print('cmat: ', self.cmat)
        print('dvec: ', self.dvec)
        print('rvecs: ', self.rvecs)
        print('tvecs: ', self.tvecs)

# Test the class ChessboardCalibration
if __name__ == '__main__':
    cc = ChessboardCalibration()
    cc.work_dir = 'd:/yuansen/impro/improMeasure/examples/chessboard_photos/'
    cc.num_corners = (13, 12)
    cc.square_size = 50.0
    cc.refresh_photo_files()
    cc.find_corners()
    while True:
        for i in range(len(cc.photo_files)):
            base_name = os.path.basename(cc.photo_files[i])
            print('%2d: %s' % (i+1, base_name))
            photo_index = i
#        photo_index = int(input('Enter the photo index (1-%d): ' % len(cc.photo_files))) - 1
            if photo_index < 0 or photo_index >= len(cc.photo_files):
                break
            cc.imshow_corners(photo_index)
        break
    cc.calibrate_camera()    

    print(cc)
    print('Done.')

        


