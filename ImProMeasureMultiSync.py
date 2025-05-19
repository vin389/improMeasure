import numpy as np
import cv2
import os, sys, glob, re, time, copy, math
import matplotlib.pyplot as plt


# date of initial creatation: 2025-04-25

class ImProMeasureMultiSync:
    # This class is used to measure the 3D coordinates of points in a multi-camera system.
    # It synchronizes the cameras and calculates the 3D coordinates of points from multiple camera images.
    # It uses the triangulation method to calculate the 3D coordinates of points from multiple camera images.

    def __init__(self, nCams=0, nPoints=0, wdir=''):
        # working directory
        self.wdir = wdir  # working directory
        # camera parameters
        # each camera saves its parameters in a file named camera_params_<icam>.txt
        # For example, camera_params_1.txt, camera_params_2.txt, etc.
        self.nCams = nCams   # number of cameras
        self.nPoints = nPoints   # number of points of interests (POIs)
        self.camImgSizes =[]  # list of camera image sizes, (width, height)
        self.rvecs = []  # list of rotational vectors
        self.tvecs = []  # list of translational vectors
        self.cmats = []  # list of 3-by-3 camera matrix
        self.dvecs = []  # list of distortion coefficients
        # video sources (or image sequences) for each camera
        # Each camera has its own video source (or image sequence) 
        # and its information is saved in a file named camera_sources_<icam>.txt
        # For example, camera_source_1.txt, camera_sources_2.txt, etc.
        self.videoSources = []  # list of video sources (or image sequences) for each camera
                                # Each element can be a string (file name) or a list of strings (file names).
                                # For a video file, the string is the file name of the video (full path).
                                # For example, self.videoSources[0] = 'd:/temp/test.mp4'
                                # And camera_source_1.txt will contain d:/temp/test.mp4 
                                # Lines starting with # are ignored.
                                # For an image sequence, it should be a list of file names (full path). 
                                # For example, self.videoSources[1] = ['d:/temp/test1.png', 'd:/temp/test2.png', 'd:/temp/test3.png']
                                # And camera_source_2.txt will contain d:/temp/test1.png\n d:/temp/test2.png\n d:/temp/test3.png\n
                                # Lines starting with # are ignored.

        # image points
        self.imgPointsAll = []  # image points of all POIs over all videos (or image sequences)


        # if given nCams is greater than 0, then initialize the related variables
        self.allocateMemory(nCams=nCams, nPoints=nPoints)

    def create_wdir(self):
        # create a working directory for the camera parameters
        # The working directory is the directory where project data is saved.
        # The working directory is created if it does not exist.
        try:
            if not os.path.exists(self.wdir):
                os.makedirs(self.wdir)    
        except OSError as e:
            print("# Error: %s - %s." % (e.filename, e.strerror))
            return False

    def allocateMemory(self, nCams, nPoints):
        if nCams < 0:
            pass
        elif (nCams == 0):
            self.camImgSizes = []
            self.rvecs = []
            self.tvecs = []
            self.cmats = []
            self.dvecs = []
            self.imgPointsAll = []
            self.videoSources = []
        else:            
            # initialize the list of camera image sizes. By default, all cameras have the same image size.
            # The image size is (width, height) in pixels.
            self.camImgSizes = [(3840, 2160)] * nCams 
            # initialize the list of rotational and translational vectors. By default, all zeros.
            self.rvecs = [np.zeros((3,1),dtype=float)] * nCams
            self.tvecs = [np.zeros((3,1),dtype=float)] * nCams
            # initialize the list of camera matrices. By default, assuming horizontal FOV is 90 degrees, 
            # and principal point is at the center of the image.
            # The camera matrix is a 3-by-3 matrix. The first two rows are the focal lengths in pixels.
            self.cmats = [np.array([ [1920,0,(3840-1.)/2.],
                                     [0,1920,(2160-1.)/2.],
                                     [0,0,1]],dtype=float)] * nCams
            # initialize the list of distortion coefficients. By default, all zeros.
            self.dvecs = [np.zeros((5,1),dtype=float)] * nCams
            # initialize the list of image points. By default, all zeros.
            # imgPointsAll is a list of nCams lists. Each list contains nPoints lists.
            # Each list contains a nSteps-by-2 numpy array. 
            # That is, imgPointsAll[icam][ipoint] is a nSteps-by-2 numpy array.
            # imgPointsAll[icam][ipoint][istep][0] is the x coordinate of camera icam, point ipoint, step istep.
            # imgPointsAll[icam][ipoint][istep][1] is the y coordinate of camera icam, point ipoint, step istep.
            # len(imgPointsAll) should be nCams.
            # len(imgPointsAll[icam]) should be number of POIs and should be the same for all cameras. 
            # The number of steps varies by camera.
            self.imgPointsAll = [ [np.zeros((0,2),dtype=float)] * nPoints for _ in range(nCams) ]
            self.videoSources = [None] * nCams  # video sources need to be defined separately. Default is None.

    def loadData(self):
        self.loadCameras()
        pass

    def saveData(self):
        self.saveCameras()
        pass

    def setCamImgSize(self, iCam, camImgSize):
        # set the image size of camera iCam
        if (iCam < 0) or (iCam >= self.nCams):
            print("Error: camera index out of range")
            return False
        self.camImgSizes[iCam] = camImgSize

    def loadCamera(self, icam):
        # load camera parameters from a file
        # The file should be in the format single column format: (w h rvec(3) tvec(3) cmat(3x3=9) dvec(5-8)
        # That is, the file should contain the following lines:
        # width of image in pixels
        # height of image in pixels
        # x of rvec vector 
        # y of rvec vector
        # z of rvec vector
        # x of tvec vector
        # y of tvec vector
        # z of tvec vector
        # c11 of cmat matrix (i.e., cmats[icam][0][0])
        # c12 of cmat matrix (i.e., cmats[icam][0][1])
        # c13 of cmat matrix (i.e., cmats[icam][0][2])
        # c21 of cmat matrix (i.e., cmats[icam][1][0])
        # c22 of cmat matrix (i.e., cmats[icam][1][1])
        # c23 of cmat matrix (i.e., cmats[icam][1][2])
        # c31 of cmat matrix (i.e., cmats[icam][2][0])
        # c32 of cmat matrix (i.e., cmats[icam][2][1])
        # c33 of cmat matrix (i.e., cmats[icam][2][2])
        # k1 of dvec vector (i.e., dvecs[icam][0])
        # (the following lines are optional:)
        # k2 of dvec vector (i.e., dvecs[icam][1])
        # p1 of dvec vector (i.e., dvecs[icam][2])
        # p2 of dvec vector (i.e., dvecs[icam][3])
        # k3 of dvec vector (i.e., dvecs[icam][4])
        # k4 of dvec vector (i.e., dvecs[icam][5])
        # k5 of dvec vector (i.e., dvecs[icam][6])
        # k6 of dvec vector (i.e., dvecs[icam][7])
        file = os.path.join(self.wdir, "camera_params_%d.txt" % (icam+1))
        with open(file, 'r') as f:
            lines = f.readlines()
            # if the first character of a line is a #, then skip the line
            lines = [line for line in lines if line[0] != '#']
            # check minimum number of lines in the file
            if len(lines) < 21:
                print("Error: file format is not correct, because the file is supposed to have at least 21 lines:")
                print("width, height, rvec(3), tvec(3), cmat(3x3=9), dvec(4-8)")
                return False
            # read the image size
            w = int(float(lines[0].strip())+.5)
            h = int(float(lines[1].strip())+.5)
            self.camImgSizes[icam] = (w, h)
            # read the rotation vector
            rvec = np.array([float(lines[2].strip()), float(lines[3].strip()), float(lines[4].strip())], dtype=float)
            self.rvecs[icam] = rvec.reshape((3,1))
            # read the translation vector
            tvec = np.array([float(lines[5].strip()), float(lines[6].strip()), float(lines[7].strip())], dtype=float)
            self.tvecs[icam] = tvec.reshape((3,1))
            # read the camera matrix
            cmat = np.zeros((3,3), dtype=float)
            for i in range(3):
                for j in range(3):
                    cmat[i][j] = float(lines[i*3+j+8].strip())
            self.cmats[icam] = cmat
            # read the distortion vector. The length of the vector is 4-8, depending on 
            # the lines in the file.
            dvec = np.zeros((8,1), dtype=float)
            for i in range(8):
                if (i+17) < len(lines):
                    dvec[i][0] = float(lines[i+17].strip())
                else:
                    break
            # set the length of the distortion vector to the number of lines read
            dvec = dvec[:i+1]
            self.dvecs[icam] = dvec.reshape((-1,1))

    def loadCameras(self):
        # load cameras
        # We do not know how many cameras are there, so we need to load the camera parameters from the files.
        # The files are named camera_params_<icam>.txt, where <icam> is the camera index.
        self.nCams = 0
        while True:
            # check if the file camera_params_<icam>.txt exists
            file = os.path.join(self.wdir, "camera_params_%d.txt" % (self.nCams+1))
            if not os.path.exists(file):
                break
            self.nCams += 1
        # allocate memory for the cameras
        self.allocateMemory(nCams=self.nCams, nPoints=self.nPoints)
        # load the cameras
        for icam in range(self.nCams):
            # load camera parameters from a file
            file = os.path.join(self.wdir, "camera_params_%d.txt" % (icam+1))
            self.loadCamera(icam)

    def saveCamera(self, icam, file=''):
        # save camera parameters to a file
        # The file will be in the format single column format: (w h rvec(3) tvec(3) cmat(3x3=9) dvec(5-8)
        # That is, the file will contain the following lines:
        # width of image in pixels
        # height of image in pixels
        # x of rvec vector 
        # y of rvec vector
        # z of rvec vector
        # x of tvec vector
        # y of tvec vector
        # z of tvec vector
        # c11 of cmat matrix (i.e., cmats[icam][0][0])
        # c12 of cmat matrix (i.e., cmats[icam][0][1])
        # c13 of cmat matrix (i.e., cmats[icam][0][2])
        # c21 of cmat matrix (i.e., cmats[icam][1][0])
        # c22 of cmat matrix (i.e., cmats[icam][1][1])
        # c23 of cmat matrix (i.e., cmats[icam][1][2])
        # c31 of cmat matrix (i.e., cmats[icam][2][0])
        # c32 of cmat matrix (i.e., cmats[icam][2][1])
        # c33 of cmat matrix (i.e., cmats[icam][2][2])
        # k1 of dvec vector (i.e., dvecs[icam][0])
        # (the following lines are optional:)
        # k2 of dvec vector (i.e., dvecs[icam][1])
        # p1 of dvec vector (i.e., dvecs[icam][2])
        # p2 of dvec vector (i.e., dvecs[icam][3])
        # k3 of dvec vector (i.e., dvecs[icam][4])
        # k4 of dvec vector (i.e., dvecs[icam][5])
        # k5 of dvec vector (i.e., dvecs[icam][6])
        # k6 of dvec vector (i.e., dvecs[icam][7])
        # if the file is not given or empty, then use the default file name
        # The default file name is camera_params<icam>.txt, where <icam> is the camera index.
        if file == '':
            file = os.path.join(self.wdir, "camera_params_%d.txt" % (icam+1))
        # if the file does not have a directory, then use the working directory
        if os.path.dirname(file) == '':
            file = os.path.join(self.wdir, file)
        # if the file already exists, then delete it
        with open(file, 'w') as f:
            f.write("# camera parameters for camera %d\n" % icam)
            f.write("# width and height of image in pixels\n")
            f.write("%d\n" % self.camImgSizes[icam][0])
            f.write("%d\n" % self.camImgSizes[icam][1])
            f.write("# rotation vector (rvec)\n")
            f.write("%f\n" % self.rvecs[icam].flatten()[0])
            f.write("%f\n" % self.rvecs[icam].flatten()[1]) 
            f.write("%f\n" % self.rvecs[icam].flatten()[2])
            f.write("# translational vector (tvec)\n")
            f.write("%f\n" % self.tvecs[icam].flatten()[0]) 
            f.write("%f\n" % self.tvecs[icam].flatten()[1])
            f.write("%f\n" % self.tvecs[icam].flatten()[2])
            f.write("# camera matrix (3x3) (cmat)\n")
            f.write("%f\n" % self.cmats[icam][0][0])
            f.write("%f\n" % self.cmats[icam][0][1])
            f.write("%f\n" % self.cmats[icam][0][2])
            f.write("%f\n" % self.cmats[icam][1][0])
            f.write("%f\n" % self.cmats[icam][1][1])
            f.write("%f\n" % self.cmats[icam][1][2])
            f.write("%f\n" % self.cmats[icam][2][0])
            f.write("%f\n" % self.cmats[icam][2][1])
            f.write("%f\n" % self.cmats[icam][2][2])
            f.write("# distortion coefficients (4x1 to 8x1) (dvec)\n")
            f.write("%f\n" % self.dvecs[icam].flatten()[0])
            f.write("%f\n" % self.dvecs[icam].flatten()[1])
            f.write("%f\n" % self.dvecs[icam].flatten()[2])
            f.write("%f\n" % self.dvecs[icam].flatten()[3])
            if len(self.dvecs[icam]) > 4:
                f.write("%f\n" % self.dvecs[icam].flatten()[4])
            if len(self.dvecs[icam]) > 5:
                f.write("%f\n" % self.dvecs[icam].flatten()[5])
            if len(self.dvecs[icam]) > 6:
                f.write("%f\n" % self.dvecs[icam].flatten()[6])
            if len(self.dvecs[icam]) > 7:
                f.write("%f\n" % self.dvecs[icam].flatten()[7])

    def saveCameras(self):
        # save cameras
        for icam in range(self.nCams):
            # save camera parameters to a file
            self.saveCamera(icam)

    def loadVideoSource(self, icam):
        # load video source from a file
        # The file should be in the format: (one line for each video source)
        # If the video source if camera icam is a video file, then the line should be:
        # d:/temp/test.mp4 
        # If the video source if camera icam is an image sequence, then the line should be:
        # d:/temp/test1.png
        # d:/temp/test2.png
        # d:/temp/test3.png
        # Lines starting with # are ignored.
        file = os.path.join(self.wdir, "camera_sources_%d.txt" % (icam+1))
        with open(file, 'r') as f:
            lines = f.readlines()
            # if the first character of a line is a #, then skip the line
            lines = [line for line in lines if line[0] != '#']
            self.videoSources[icam] = [line.strip() for line in lines]

    def createVideoSourceFileListByCStyleSpecifier(self, icam, fileName_pattern, startIndex=1, nFiles=100, step=1):
        # create a list of file names for the video source
        # The file name should be in the format: d:/temp/test1_%d.png, where %d is the C-style specifier, can be %d, %6d, or %06d styles.
        # The %04d means that the number is 4 digits long and is padded with zeros.
        # For example, if the file name is d:/temp/test1_%04d.png, then the file names are:
        # d:/temp/test1_0001.png, d:/temp/test1_0002.png, d:/temp/test1_0003.png, etc.
        # The number of files is determined by the number of frames in the video or image sequence.
        # The file name should be in the format: d:/temp/test1_%04d.png
        # The %04d means that the number is 4 digits long and is padded with zeros.
        # For example, if the file name is d:/temp/test1_%04d.png, then the file names are:
        # d:/temp/test1_0001.png, d:/temp/test1_0002.png, d:/temp/test1_0003.png, etc.
        # The number of files is determined by the number of frames in the video or image sequence.
        # create a text file "camera_sources_<icam>.txt" in the working directory
        with open(os.path.join(self.wdir, "camera_sources_%d.txt" % (icam+1)), 'w') as f:
            # create a list of file names for the video source
            # The file name should be in the format: d:/temp/test1_%d.png, where %d is the C-style specifier, can be %d, %6d, or %06d styles.
            # The %04d means that the number is 4 digits long and is padded with zeros.
            # For example, if the file name is d:/temp/test1_%04d.png, then the file names are:
            # d:/temp/test1_0001.png, d:/temp/test1_0002.png, d:/temp/test1_0003.png, etc.
            fileName_pattern_dir = os.path.dirname(fileName_pattern)
            fileName_pattern_base = os.path.basename(fileName_pattern)
            for i in range(startIndex, startIndex+nFiles*step, step):
                try:
                    filename_base = fileName_pattern_base % i
                except:
                    print("# Error: file name pattern is not correct.")
                    print("#        file name pattern should contain a C-style specifier, such as %d, %6d, or %06d.")
                    print("#        But now it is %s" % fileName_pattern_base)
                # for the first file, use the full path
                filename = os.path.join(fileName_pattern_dir, filename_base)
                # write 
                f.write(filename + '\n')
        pass

    def createVideoSourceFileListByFileDialog(self, icam):
        import tkinter as tk
        from tkinter import filedialog
        # create a file dialog to select the video source, that allows user to select multiple files
        root = tk.Tk()
        root.withdraw()
        # ask user to select the video source file(s).
        # that allows Video file (*.mp4;*.avi;*.mov) and Image files (*.png;*.jpg;*.jpeg;*.bmp;*.tiff,*.tif)
        filename = filedialog.askopenfilename(
            title="Select video source for camera %d (1 video or multiple images)" % (icam+1),
            filetypes=[("Video or images", "*.mp4;*.avi;*.mov;*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")],
            initialdir=self.wdir,
            multiple=True
        )
        # if the user cancels the dialog, then return
        if not filename:
            return False
        # if file(s) are selected, write the files to camera_sources_%d.txt" % (icam+1))
        with open(os.path.join(self.wdir, "camera_sources_%d.txt" % (icam+1)), 'w') as f:
            for file in filename:
                f.write(file + '\n')
        print("# Selected %d files for camera %d." % (len(filename), icam+1))
        pass       
        




def test_01():
    # test the class
    wdir = r'D:\yuansen\ImPro\improMeasure\tests\test_dir_ImProMeasureMultiSync_v0'
    imPro = ImProMeasureMultiSync(nCams=2, nPoints=3, wdir=wdir)
#    imPro.saveData(wdir)
    imPro.loadData()
    imPro_verify = copy.deepcopy(imPro)
    imPro_verify.wdir = wdir+'_verify'
    imPro_verify.saveData()

def test_02():
    wdir = r'D:\yuansen\ImPro\improMeasure\tests\test_dir_ImProMeasureMultiSync_v0'
    imPro = ImProMeasureMultiSync(nCams=2, nPoints=3, wdir=wdir)
    imPro.createVideoSourceFileListByCStyleSpecifier(0, 
          r'D:\yuansen\ImPro\improMeasure\tests\test_dir_ImProMeasureMultiSync_v0\test1_%04d.png', 
          startIndex=1, nFiles=10, step=2)
    pass

def test_03_createVideoSourceFileListByFileDialog():
    wdir = r'D:\yuansen\ImPro\improMeasure\tests\test_dir_ImProMeasureMultiSync_v0'
    imPro = ImProMeasureMultiSync(nCams=2, nPoints=3, wdir=wdir)
    imPro.createVideoSourceFileListByFileDialog(icam=0)
    imPro.createVideoSourceFileListByFileDialog(icam=1)
    pass

def test_04_b4_test():
    wdir = r'D:\yuansen\ImPro\improMeasure\tests\test_dir_ImProMeasureMultiSync_v0'
    vfiles = [r'D:\ExpDataSamples\20250300_B4_shake_table_tests\sample_data_1\exp1_xydirection_0.5hz\gopro\gopro7_xy_0.5hz.MP4',\
              r'D:\ExpDataSamples\20250300_B4_shake_table_tests\sample_data_1\exp1_xydirection_0.5hz\gopro\gopro121_xy_0.5hz.MP4',\
              r'D:\ExpDataSamples\20250300_B4_shake_table_tests\sample_data_1\exp1_xydirection_0.5hz\gopro\gopro122_xy_0.5hz.MP4']
    ncam = len(vfiles)
    calib_files = [vfiles[0].replace('.MP4', '_calib_k1_k2_p1_p2.txt'),\
                   vfiles[1].replace('.MP4', '_calib_k1_k2_p1_p2.txt'),\
                   vfiles[2].replace('.MP4', '_calib_k1_k2_p1_p2.txt')]
    poi_files = [os.path.join(wdir, 'poi_cam_%02d.txt' % (i+1)) for i in range(ncam)]


    pass

    imPro = ImProMeasureMultiSync(nCams=2, nPoints=3, wdir=wdir)


if __name__ == "__main__":
    # test the class
    # test_01()
    # test_02()
    # test_03_createVideoSourceFileListByFileDialog()
    test_04_b4_test()


    pass
    pass



