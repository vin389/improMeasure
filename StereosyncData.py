import os
import sys
import time
import numpy as np
import cv2
import scipy

from inputs import input2
from pickTemplates import pickTemplates
from Camera import Camera

from eccTrackVideo_v3 import eccTrackVideo_v3
from syncByTriangulation import syncByTriangulation
from triangulatePoints2 import triangulatePoints2

# rootWin, workDir, fVideos[0:2], fCalibs[0:2], fTmplts[0:2],
# videos[0:2], videoResolutions[0:2], videoFrameCounts[0:2], videoFpss[0:2],
# tmpltFrameIds[0:2], imgInits[0:2],
# cmats[0:2], dvecs[0:2], rvecs[0:2], tvecs[0:2],
# trackCams[cam ids], trackPoints[point ids], trackFrameRange[frame ids],
# lagTrials[lags],

class StereosyncData():
    def __init__(self,
                 workDir='',
                 fVideos=['', ''],
                 fCalibs=['', ''],
                 fTmplts=['', ''],
                 tmpltFrameIds=np.array([0, 0]),
                 trackCams=np.array([0, 1]),
                 trackPoints=np.array([0,1,2,3], dtype=int),
                 trackFrameRange=np.array([[0,9999],[0,9999]],dtype=int),
                 lagTrialRange=np.array([-60,61], dtype=int),
                 syncPoints=np.array([0,1,2,3], dtype=int),
                 syncFrameRange=np.array([120,9999-120], dtype=int),
                 triangulationPoints=np.array([0,1,2,3], dtype=int),
                 triangulationFrameRange=np.array([120,9999-120], dtype=int),
                 ):
        """
        StereosyncData is a class that is designed to manage the basic data for
        tkStereosync_v3. The tkStereosync is a series of programs that 
        are prototyped to investigate the stereo synchronization. The stereo 
        synchronization is a method that does synchronization between two 
        un-synchronized videos (which are very common when we use consumer 
        cameras).
        
        When using StereosyncData (no GUI) or tkStereosync (with Tk interface),
        user needs to assign the following information:
            
        * working directory (workDir), where user prefer to save the analyzed 
          data and configuration of this analysis. If users want to do multiple
          trials of analysis, they need to have multiple directories, one  
          directory each analysis. For example, "c:/data/mydir" in Windows or
          "/data/mydir" in unix based OS like Linux or MacOS.
          
        * files of videos (fVideos), which are two file names of the two videos.
          They only need to include the file names without the path if the 
          files are in the working directory. However, if the video files are 
          not in the working directory, user need to use ".." (i.e., up one 
          level from a file path). For example, if the working directory is 
          "/data/mydir" while a video is at "/data/dir3/video.mp4", the file
          can be assigned to "../dir3/video.mp4"

        * files of calibration result files (fCalibs), which are two files of   
          camera parameters. The file should contain the following numeric data
          in a column in text form: w h rx ry rz tx ty tz c11 c12 c13 c21 c22 
          c23 c31 c32 c33 k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y
          w and h are numbers of horizontal and vertical pixels of the video
          camera, respectively. Parameters after k2 are optional, so you can 
          either ignore them or set them zeros. The s1 to s4 are for fish-eye 
          camera. The tau_x and tau_y are rarely documented. Read OpenCV manual
          for more details. The file can be obtained from program tkCalib.py.
          You can use '#' in the very beginning of a line in the text file to 
          indicate the line is a comment for human and is to be ignored by the 
          computer (as we use numpy loadtxt to load the file.)
          Remember: Improper camera parameters could lead to 
          unreasonable stereo triangulation result, and this program does not 
          do the check (assuming the verification of camera parameters are 
          users' responsibility'). It is not easy to describe in a few words 
          how to determine whether a set of camera parameters are reasonable. 
          It would be better for users to learn camera parameters well 
          before using this program.
          rx ry rz tx ty tz are rotational vectors and translational vectors,
          respectively, of the extrinsic paramters of the video camera.
          c11 to c33 are the 3x3 camera matrix, where c11 and c22 are the focal
          lengths (in unit of pixel), c13 and c23 are the principal points (in
          unit of pixel). c33 is one, and c12, c21, c31, and c32 are zeros 
          typically (unless you actually know what you are doing). 

        * file of templates (fTmplts), which are two files of templates of two
          cameras. Each line of the text file represents a template, which 
          contains 6 numbers: xi yi x0 y0 w h, where xi and yi are the image 
          coordinate of the template (can be reals), while the x0, y0, w, and h
          (must be integers) are the upper-left corner and size of the template
          . They can be generated by a function pickTemplates() in 
          pickTemplates.py. 
          In the current version of this program, templates in video 1 and 
          video 2 should be corresponding. That is, the template i in both 
          videos refer to the same point (point i). 
          You can use '#' in the very beginning of a line in the text file to 
          indicate the line is a comment for human and is to be ignored by the 
          computer (as we use numpy loadtxt to load the file.)

        * two integers that are frames IDs that you want to define templates in
          the videos (tmpltFrameIds). Many of us probably consider the first 
          frame (index 0) as the initial image and use the image to define the
          images of templates. That is common. However, sometimes the first 
          few frames of a video may not be taken unstably because video cameras 
          were touched and vibrated while the shutter buttons were pressed. In 
          those cases, users may want to assign the initial frames to several 
          frames or dozens frames after video cameras start. This is why this 
          program asks users to assign the frame IDs that define the templates.
          If you are pretty sure the very first frame of two videos are what 
          to want to define templates, you assign 0 and 0 for them. 

        * camera IDs that you want to do image tracking (trackCams). Normally 
          you assign 0 and 1 (i.e., both cameras) to trackCams, as in most of 
          the cases we track targets in both cameras. However, sometimes we 
          re-define templates of one of the cameras and want to re-do image 
          tracking (for example, camera 0), and we set the camera IDs to 0 so 
          that we only re-track templates in camera 0, without changing the 
          tracking results of camera 1. 

        * point IDs that you want to do image tracking (trackPoints). Noramlly
          we assign all templates, for example if we have 20 templates in each
          of the cameras, we assign 0 1 2 3 ... 19 for trackPoints[0] and 
          0 1 2 3 ... 19 for trackPoints[1]. However, sometimes we re-define 
          only selected templates and want to re-do tracking for these selected
          templates, you can assign them specifically. 

        * frame IDs that you want to do image tracking (trackFrameRange). 
          Normally we track all frames over the entire videos. 
          For example, for a 60-second video with frame rate of 59.94 FPS (say, 
          there are 3596 frames in the video) for both cameras, we would set 
          0 3596 
          0 3596 
          to trackFrameRange[0] and trackFrameRange[1], respectively.We only  
          assign the start frame and the end index of the range, not all 
          indices of every single frame. The so-called "end" 
          is actually not included (as we use Python style of range, not 
          Matlab style of range). 
          Remember: This program assumes the tracking frames are continuous 
          frame after frame, without any skipping between start and end. (
          Again, the "end" is not included, that is, the Python style.) 
          One of the reasons is we read frames from a video frame by frame 
          without skipping. Randomly access of frames from a video file could 
          result in an unexpected result. 
          
        * the range of possible lags (lagTrialRange), for example:
          [-100, 101] 
          represents you feel the time difference between two videos are 
          at most between +/- 100 frames. (Again, "end" is not included, that 
          is Python style). A positive lag indicates that the 2nd camera (
          camera 1 is the 2nd camera, that is Python/C style) was turned on 
          later than the camera 1. For example, lag of 100 means the frame K 
          in camera 1 is taken at the same time that camera 0 took frame K+100.
          You need to give integers for lagTrialRange, while the best lag that 
          this program determines could be a real (for each point).
          
        * the range of frames for stereo synchronization (syncFrameRange), 
          for example:
          [200, 4001]
          represents frames from 200 to 4000 of the first camera (camera 0) 
          (again, "end" is not included) and their corresponding frames in the 
          other camera (camera 1) are used for stereo synchronization. 
          Note, frame index are 0 based. That is, frame 100 is the 101st frame
          of a video. 
          You need to understand the range could be a little bit narrower than
          the entire video because of the lagTrialRange. 
          For example, if lagTrialRange is [-100,101] and syncFrameRange is 
          [200, 4001], it means in case of lagTrialRange of -100 (i.e., )
          the range for synchronization is 
            [300, 4101] for camera 1 ([200, 4001] for camera 0)
          and in case of lagTrialRange of 100 (i.e., 101-1), the range is
            [100, 3901] for camera 1 ([200, 4001] for camera 0)
          
        * triangulation points (triangulationPoints), the points that user
          wants to get the 3D trajectories. These triangulation will be done
          considering the synchronized results (best lags of each point). 
          For example, if user wants to get the trajectories of all 20 points,
          the triangulation points will be 
          [0,1,2, ... 19]
          
        * triangulation frame range (triangulationFrameRange), the range of 
          frames (of camera 0) that the user wants to do triangulation. If the
          user wants to do it through the entire videos duration, the 
          triangulation frame range can be:
          [0, 99999] 
          (where the 99999 represents the maximum number of frames)
          However the actual range will be narrower because the synchronizaion.
          For example, if the video lengths are 4800 and 4600 for cameras 0 
          and 1, respectively, and the time lag of a certain point is 50.5 (
          meaning the camera 1 started recording later), the actual (widest)
          triangulation frame range would be:
          [51, 4650]
          It starts from frame 51 because at frame 50 the camera 1 has not 
          turned on and has no image. It ends at frame 4650 (as "end" is not 
          included, Python style, and the frame index 4649 of camera 0 is at
          equivalent time of frame index 4598.5) as camera 1 has only 4600 
          frames and the last frame index is 4599. 
          
        Parameters
        ----------
        workDir : TYPE string, optional
            Working directory that is to be followed by fVideos, fCalibs,
            and fTmplts. For example, 
            workDir is 'c:\\myDirectory'
            fVideos is ['video1.mp4', 'video2.mp4']
            In this program, the video file would be, for example,
            os.path.join(workDir, fVideos[0])
        fVideos : TYPE list of strings, optional
            Full paths of video files. The default is ['',''].
              For example, ['v1.mp4', 'v2.mp4']
        fCalibs : TYPE list of strings, optional
            File names of camera parameters files. The default is ['',''].
              For example, ['cam1.csv', 'cam2.csv']
                Each file contains 21 to 31 scalars: width(1), height(1), 
                rvec(3), tvec(3), cmat(9), dvec(4 to 14)
        fTmplts : TYPE list of strings, optional
            File names of templates files. The default is ['',''].
              For example, ['tmplts1.csv', 'tmplts2.csv'].
                Each file contains N rows to define N templates. 
                Each template is defined by 6 scalars: xi, yi, x_upper_left,
                y_upper_left, width, height. The xi and yi are reals. The 
                latter 4 are integers. 
        tmpltFrameIds : TYPE numpy integer array (shape: 2,), optional
            Frame indices of images that are used to define templates. 
            The default is [0,0].
            0 means that the template images are defined from the very first
            frame of the video. The templates do not need too be defined by 
            the first frame.
        trackCams : TYPE numpy integer array (shape: 2,), optional
            Camera indices to run image point tracking. The default is [0, 1].
            0-based index. 0 is camera 1. 1 is camera 2.
        trackPoints : TYPE 1D integer numpy array, optional
            trackPoints is a numpy array, which contains points to do image tracking. 
            Points are 0 based indices.
        trackFrameRange : TYPE 2x2 numpy integer array 
            trackFrameRange[0] is an np array, which contains the starting frame ([0,0]) and end frame ([0,1]) of camera 1 to track. 
            trackFrameRange[1] is an np array, which contains the starting frame ([1,0]) and end frame ([1,1]) of camera 2 to track. 
            Frames are 0 based indices.
        lagTrialRange : TYPE range of integers, optional
            range of time lags to try. The default is [-60,61], meaning the trials are -60,-59,...,59.60.
            For example: [-60,61].
        syncPoints : TYPE 1D numpy array of integers, optional
            syncPoints is a numpy array, which contains points that are used to 
              do the stereo synchronization. 
        syncFrameRange : TYPE 1D int array or list, optional
            syncFrameRange is a 1D int array which has only two integers (start and end), 
            which contains frames of camera 1 that are used to do synchronization.
            Frames for sync must be continuous, so we only need the start and end.
            The range does not include 'end' itself, i.e., the python style.
            For user interface, the range include the user's end because we assume
            our user is more comfortable with that range includes the 'end' itself,
            that is, the matlab style.
        triangulationPoints : TYPE 1D integer arrays, optional
            triangulationPoints is an np array, which contains points that are 
              used to do the stereo triangulation. 
        triangulationFrameRange : TYPE list of integers, optional
            triangulationFrameRange is a list which contain frames of camera 1
            that do triangulation (with frames of camera 2 which considers
            bestTimeLags[]. This function use interpolation to estimate the
            image coordinates of points in camera 2 as bestTimeLags[] are 
            not necessary to be integers.)
        Returns
        -------
        None.

        """
        self.workDir = workDir
        self.fVideos = fVideos.copy() # deep copy everything
        self.fCalibs = fCalibs.copy() # deep copy everything
        self.fTmplts = fTmplts.copy() # deep copy everything. 
        self.tmpltFrameIds = tmpltFrameIds.copy() # deep copy everything. 
        self.trackCams = trackCams.copy()
        self.trackPoints = trackPoints.copy() # deep copy everything
        self.trackFrameRange = trackFrameRange.copy() # deep copy everything
        self.lagTrialRange = lagTrialRange.copy()
        self.syncPoints = syncPoints.copy()
        self.syncFrameRange = syncFrameRange.copy()
        self.triangulationPoints = triangulationPoints.copy()
        self.triangulationFrameRange = triangulationFrameRange.copy()

        self.cams = [None, None]  # objects of Camera (defined in Camera.py)
        self.videos = [None, None]  # cv2.VideoCapture() objects
        # OpenCV images that define images of templates
        self.imgInits = [None, None]
        self.videoResolutions = [(0, 0), (0, 0)]
        self.videoFpss = [0., 0.]
        self.videoFrameCounts = [0, 0]
        # tmplts[icam] is an N-by-6 numpy array. Each row is a template: xi, yi, x0, y0, w, h
        self.tmplts = [None, None]

        self.bigTable = np.array([], dtype=np.float64)  # (frameCount, )
        self.prjMseTable = np.zeros((0,0), dtype=np.float64) # (np, )
        self.bestTimeLags = np.array([], dtype=np.float64) # (np,), np is number of points of interests
    # end of def __init__(self, ...)

    def __str__(self):
        theStr = []
        #  workDir
        theStr.append("# workDir: %s" % self.workDir)
        # display fVideos
        theStr.append("# fVideos 1: %s" % self.fVideos[0])
        theStr.append("# fVideos 2: %s" % self.fVideos[1])
        # display fCalibs
        theStr.append("# fCalibs 1: %s" % self.fCalibs[0])
        theStr.append("# fCalibs 2: %s" % self.fCalibs[1])
        # display fTmplts
        theStr.append("# fTmplts 1: %s" % self.fTmplts[0])
        theStr.append("# fTmplts 2: %s" % self.fTmplts[1])
        # display tmpltFrameIds
        theStr.append("# tmpltFrameIds (indices are 0-base): %s" % self.tmpltFrameIds.__str__())
        # display trackCams
        theStr.append("# trackCams (indices are 0-base): %s" % self.trackCams.__str__())
        # display trackPoints
        theStr.append("# trackPoints (indices are 0-base): %s" % self.trackPoints.__str__())
        # display trackFrameRange
        theStr.append("# trackFrameRange 1 (indices are 0-base, end is not included, Python style): %s" % self.trackFrameRange[0].__str__())
        theStr.append("# trackFrameRange 2 (indices are 0-base, end is not included, Python style): %s" % self.trackFrameRange[1].__str__())
        # display lagTrialRange
        theStr.append("# lagTrialRange: %s" % self.lagTrialRange.__str__())
        # display syncPoints
        theStr.append("# syncPoints (indices are 0-base): %s" % (self.syncPoints[0]).__str__())
        # display syncFrameRange
        theStr.append("# syncFrameRange (indices are 0-base. end is not included, Python style): %s" % self.syncFrameRange.__str__())
                                   # the "-1" converts "python-style end (not included)" to "matlab-style end (included)". 
        # display triangulationPoints
        theStr.append("# triangulationPoints (indices are 0-base): %s" % (self.triangulationPoints).__str__())
        # display triangulationFrameRange
        theStr.append("# triangulationFrameRange (indices are 0-base. end is not included, Python style): %s" % self.triangulationFrameRange.__str__()) 
                                   # the "-1" converts "python-style end (not included)" to "matlab-style end (included)". 
        # display cams (camera parameter objects)
        for icam in range(2):
            if type(self.cams[icam]) != Camera:
                theStr.append("# cams %d (camera parameters objects): is not created." % (icam+1))
            else:
                theStr.append("# cams %d position: %s" % (icam+1, self.cams[icam].campos().flatten()))
        # display videos (cv2.VideoCapture objects)
        for icam in range(2):
            if type(self.videos[icam]) != cv2.VideoCapture:
                theStr.append("# videos %d: not created." % (icam+1))
            elif self.videos[0].isOpened():
                theStr.append("# videos %d: created and opened." % (icam+1))
            else:
                theStr.append("# videos %d: created but not opened." % (icam+1))
        # display imgInits
        for icam in range(2):
            if type(self.imgInits) != list or len(self.imgInits) < 1 or type(self.imgInits[0]) != np.ndarray:
                theStr.append("# imgInit %d: image is not loaded yet." % (icam+1))
            else:
                theStr.append("# imgInit %d shape: %s" % (icam+1, self.imgInits[icam].shape))
        # display videoResolutions
        for icam in range(2):
            theStr.append("# videoResolutions %d: %s" % (icam+1, self.videoResolutions[icam]))
        # display videoFpss
        for icam in range(2):
            theStr.append("# videoFpss %d: %f" % (icam+1, self.videoFpss[icam]))
        # display videoFrameCounts   
        for icam in range(2):
            theStr.append("# videoFrameCounts %d: %d" % (icam+1, self.videoFrameCounts[icam]))
        # display tmplts
        for icam in range(2):
            if type(self.tmplts[icam]) == type(None):
                theStr.append("# tmplts of video %d: Not defined yet." % (icam+1))
            else:
                theStr.append("# tmplts of video %d: There are %d templates defined." % (icam+1, self.tmplts[icam].shape[0]))
        # display bigTable shape
        if type(self.bigTable) == np.ndarray:
            theStr.append("# bigTable.shape is %s" % str(self.bigTable.shape))
        else:
            theStr.append("# bigTable.shape is not assigned or type is not np.ndarray")
        # display prjMseTable shape
        if type(self.prjMseTable) == np.ndarray:
            theStr.append("# prjMseTable.shape is %s" % str(self.prjMseTable.shape))
        else:
            theStr.append("# prjMseTable.shape is not assigned or type is not np.ndarray")
        # display bestTimeLags shape
        if type(self.bestTimeLags) == np.ndarray:
            theStr.append("# bestTimeLags.shape is %s" % str(self.bestTimeLags.shape))
        else:
            theStr.append("# bestTimeLags.shape is not assigned or type is not np.ndarray")
        return '\n'.join(theStr)

    def __repr__(self):
        return self.__str__()

    def setWorkDir(self, workDir):
        if type(workDir) == str:
            self.workDir = workDir
        if os.path.isdir(workDir) == False:
            print("# Warning: Working directory %s is not a directory."
                  % workDir)
        if os.path.exists(workDir) == False:
            print("# Warning: Working directory %s does not exist."
                  % workDir)
    # end of def setWorkDir(self, workDir):

    def videoFile(self, icam):
        if type(self.fVideos) != list:
            print("# Error: Video files are not set.")
            return ""
        if len(self.fVideos) < icam+1:
            print("# Error: Video file %d is not set." % (icam+1))
        if type(self.fVideos[icam]) != str:
            print("# Error: Video file %d is not set." % (icam+1))
        if type(self.workDir) != str:
            print("# Warning: Working directory (workDir) is not set.")
            self.workDir = ''
        if len(self.workDir) <= 0:
            print("# Warning: Working directory (workDir) is empty.")
        return os.path.join(self.workDir, self.fVideos[icam])
    # end of def videoFile(self, icam):
        
    def calibFile(self, icam):
        if type(self.fCalibs) != list:
            print("# Error: Calibration files are not set.")
            return ""
        if len(self.fCalibs) < icam+1:
            print("# Error: Calibration file %d is not set." % (icam+1))
        if type(self.fCalibs[icam]) != str:
            print("# Error: Calibration file %d is not set." % (icam+1))
        if type(self.workDir) != str:
            print("# Warning: Working directory (workDir) is not set.")
            self.workDir = ''
        if len(self.workDir) <= 0:
            print("# Warning: Working directory (workDir) is empty.")
        return os.path.join(self.workDir, self.fCalibs[icam])
    # end of def calibFile(self, icam):
        
    def tmpltFile(self, icam):
        if type(self.fTmplts) != list:
            print("# Error: Template files are not set.")
            return ""
        if len(self.fTmplts) < icam+1:
            print("# Error: Template file %d is not set." % (icam+1))
        if type(self.fTmplts[icam]) != str:
            print("# Error: Template file %d is not set." % (icam+1))
        if type(self.workDir) != str:
            print("# Warning: Working directory (workDir) is not set.")
            self.workDir = ''
        if len(self.workDir) <= 0:
            print("# Warning: Working directory (workDir) is empty.")
        return os.path.join(self.workDir, self.fTmplts[icam])
    # end of def tmpltFile(self, icam):


    def loadVideoInfo(self, icam, printInfo=True):
        self.loadVideo(icam, printInfo)
        if type(self.videos[icam]) == cv2.VideoCapture:
            self.videos[icam].release()
        if printInfo:
            print()
    # end of def loadVideoInfo(self, icam, printInfo=True):

    def setVideoFile(self, icam, fVideo, printInfo=True):
        self.fVideos[icam] = fVideo
        self.loadVideoInfo(icam, printInfo)
    # end of def setVideoFile(self, icam, fVideo, printInfo=True):

    def loadCalibFile(self, icam, printInfo=True):
        fCalib = os.path.join(self.workDir, self.fCalibs[icam])
        fileExists = os.path.exists(fCalib)
        if printInfo:
            print("# The camera parameters file existence is %s" % fileExists)
        if fileExists:
            # load calibration parameters
            try:
                self.cams[icam] = Camera()
                self.cams[icam].loadFromFile(fCalib)
                if printInfo:
                    print("# Camera position: ",
                          self.cams[icam].campos().flatten())
            except:
                print("# Failed to load camera calibration file %s" % fCalib)
        # end of if fileExists:
        if printInfo:
            print()
    # end of def loadCalibFile(self, icam, printInfo=True):

    def setCalibFile(self, icam, fCalib, printInfo=True):
        self.fCalibs[icam] = fCalib
        self.loadCalibFile(icam, printInfo)
    # end of def setCalibFile(self, icam, fCalib, printInfo=True):

    def loadTmpltFile(self, icam, printInfo=True):
        fTmplt = os.path.join(self.workDir, self.fTmplts[icam])
        fileExists = os.path.exists(fTmplt)
        if printInfo:
            print("# The template file existence is %s" % fileExists)
        if fileExists:
            # load templates
            try:
                self.tmplts[icam] = np.loadtxt(fTmplt, delimiter=',')
                if self.tmplts[icam].shape[1] != 6:
                    print(
                        "# Warning: Each template should be defined by 6 numbers. Check the file %s." % fTmplt)
                else:
                    print("# Loaded %d templates from %s"
                          % (self.tmplts[icam].shape[0], fTmplt))
                    print("# The first template is ", self.tmplts[icam][0])
                    print("# The last template is ", self.tmplts[icam][-1])
                # end of if self.tmplts[icam].shape[1] != 6:
            except:
                print("# Failed to load template file %s" % fTmplt)
        # end of if fileExists:
        if printInfo:
            print()
    # end of def loadTmpltFile(self, icam, printInfo=True):

    def setTmpltFile(self, icam, fTmplt, printInfo=True):
        self.fTmplts[icam] = fTmplt
        self.loadTmpltFile(icam, printInfo)
    # end of def setTmpltFile(self, icam, fTmplt, printInfo=True):

    def setTmpltFrameIds(self, icam, tmpltFrameId, printInfo=True):
        self.tmpltFrameIds[icam] = int(tmpltFrameId)
        if printInfo:
            print("# Set cam %d template frame to %d" %
                  (icam+1, tmpltFrameId+1))
        pass
    # end of def settmpltFrameIds(self, icam, tmpltFrameIds)

    def setTrackCams(self, trackCams):
        try:
            trackCams = np.array(trackCams,dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setTrackCams(): trackCams should be [0], [1], or [0,1].")
            return
        if trackCams.size > 2 or min(trackCams) < 0 or max(trackCams) > 1:
            print("# Error. StereosyncData.setTrackCams(): trackCams should be [0], [1], or [0,1].")
            return
        self.trackCams = trackCams
    # end of setTrackCams(self, trackCams)
    
    def setTrackPoints(self, trackPoints):
        try:
            trackPoints = np.array(trackPoints,dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setTrackPoints(): trackPoints should be a list (or np array) of points you want to track. For example, [0,1,2,..., 19]")
            return
        if len(trackPoints) > min(self.nPoints()):
            print("# Error. StereosyncData.setTrackPoints(): Length of trackPoints (%d) should not be longer than number of points (%d)." % (len(trackPoints), self.nPoints().__str__()))
            return
        if min(trackPoints) < 0:
            print("# Error. StereosyncData.setTrackPoints(): trackPoints cannot contain negative value.")
            return
        if max(trackPoints) >= min(self.nPoints()):
            print("# Error. StereosyncData.setTrackPoints(): trackPoints cannot contain point(s) that does not exist (%d)." % max(trackPoints))
            return 
        self.trackPoints = trackPoints
    # end of setTrackPoints(self, trackPoints)
    
    def setTrackFrameRange(self, icam, theRange):
        if icam != 0 and icam != 1:
            print("# Error. StereosyncData.setTrackFrameRange: icam should be either 0 or 1.")
            return
        try:
            theRange = np.array(theRange, dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setTrackFrameRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        self.trackFrameRange[icam,0] = max(0, theRange[0])
        self.trackFrameRange[icam,1] = min(self.nFrames()[icam], theRange[1])
        print("# The trackFrameRange[%d] (index is 0 base) is set to %s" % (icam, self.trackFrameRange[icam].__str__()))
    # end of def setTrackFrameRange(icam, the Range):

    def setLagTrialRange(self, theRange):
        try:
            theRange = np.array(theRange, dtype=int)
        except:
            print("# Error. StereosyncData.setLagTrialRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        if theRange.size != 2:
            print("# Error. StereosyncData.setLagTrialRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        self.lagTrialRange = theRange
        pass
#   end of def setLagTrialRange(self, lagTrialRange):

    def setSyncPoints(self, syncPoints):
        try:
            syncPoints = np.array(syncPoints,dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setSyncPoints(): syncPoints should be a list (or np array) of points you want to do synchronization. For example, [0,1,2,..., 19]")
            return
        if len(syncPoints) > min(self.nPoints()):
            print("# Error. StereosyncData.setSyncPoints(): Length of syncPoints (%d) should not be longer than number of points (%d)." % (len(syncPoints), self.nPoints().__str__()))
            return
        if min(syncPoints) < 0:
            print("# Error. StereosyncData.setSyncPoints(): syncPoints cannot contain negative value.")
            return
        if max(syncPoints) >= min(self.nPoints()):
            print("# Error. StereosyncData.setSyncPoints(): syncPoints cannot contain point(s) that does not exist (%d)." % max(syncPoints))
            return 
        self.syncPoints = syncPoints
    # end of setSyncPoints(self, trackPoints)

    def setSyncFrameRange(self, theRange):
        try:
            theRange = np.array(theRange, dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setSyncFrameRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        if theRange.size != 2:
            print("# Error. StereosyncData.setSyncFrameRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        if self.nFrames()[0] <= 0 or self.nFrames()[1] <= 0:
            print("# Warning. StereosyncData.setSyncFrameRange: # of video frames are zeros. I suggest you to set the videos first before setting sync. frame range.")
        self.syncFrameRange[0] = max(0, theRange[0])
        self.syncFrameRange[1] = min(min(self.nFrames()), theRange[1])
        print("# The syncFrameRange is set to %s" % (self.syncFrameRange.__str__()))
    # end of def setSyncFrameRange(self, theRange):

    def setTriangulationPoints(self, triagPoints):
        try:
            triagPoints = np.array(triagPoints,dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setTriangulationPoints(): triagPoints should be a list (or np array) of points you want to do triangulation. For example, [0,1,2,..., 19]")
            return
        if len(triagPoints) > min(self.nPoints()):
            print("# Error. StereosyncData.setTriangulationPoints(): Length of triagPoints (%d) should not be longer than number of points (%d)." % (len(triagPoints), self.nPoints().__str__()))
            return
        if min(triagPoints) < 0:
            print("# Error. StereosyncData.setTriangulationPoints(): triagPoints cannot contain negative value.")
            return
        if max(triagPoints) >= min(self.nPoints()):
            print("# Error. StereosyncData.setTriangulationPoints(): triagPoints cannot contain point(s) that does not exist (%d)." % max(triagPoints))
            return 
        self.triangulationPoints = triagPoints
    # end of setTriangulationPoints(self, triagPoints)

    def setTriangulationFrameRange(self, theRange):
        try:
            theRange = np.array(theRange, dtype=int).flatten()
        except:
            print("# Error. StereosyncData.setTriangulationFrameRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        if theRange.size != 2:
            print("# Error. StereosyncData.setTriangulationFrameRange: theRange should be a list (or np array) that contains only two elements (i.e., the start and end.)")
            return
        self.triangulationFrameRange[0] = max(0, theRange[0])
        self.triangulationFrameRange[1] = min(min(self.nFrames()), theRange[1])
        print("# The triangulationFrameRange is set to %s" % (self.triangulationFrameRange.__str__()))
    # end of def setTriangulationFrameRange(the Range):

    def checkSyncFrameRange(self):
        print("# The number of frames for two videos are %d and %d, respectively." % (self.nFrames()[0], self.nFrames()[1]))
        print("# The current lag trial range is [ %d , %d ] (end is not included)." % (self.lagTrialRange[0], self.lagTrialRange[1]))
        print("# The current sync frame range is [ %d , %d ] (end is not included)." % (self.syncFrameRange[0], self.syncFrameRange[1]))
        #
        newSyncFrameRange = self.syncFrameRange.copy()
        #
        print("# If lag is %d:" % self.lagTrialRange[0])
        lagTrial = self.lagTrialRange[0]
        print("#    Sync frames in video 0 are from %d to %d" % (newSyncFrameRange[0], newSyncFrameRange[1]-1))
        if newSyncFrameRange[0] < 0:
            print("# We need to modify sync-frame-range start from %d to 0." % newSyncFrameRange[0])
            newSyncFrameRange[0] = 0
        if newSyncFrameRange[1] > self.nFrames()[0]:
            print("# We need to modify sync-frame-range end frame from %d to %d due to limited length of video 0." % (newSyncFrameRange[1], self.nFrames()[0]))
            newSyncFrameRange[1] = self.nFrames()[0]
        print("#    Sync frames in video 1 are from %d to %d" % \
              (newSyncFrameRange[0] - lagTrial, \
               newSyncFrameRange[1]-1 - lagTrial))
        if newSyncFrameRange[0] - self.lagTrialRange[0] < 0:
            print("# We need to modify sync-frame-range start frame from %d to %d due to start frame of video 1." % \
                  (newSyncFrameRange[0], lagTrial))
            newSyncFrameRange[0] = lagTrial
        if newSyncFrameRange[1]-1 - lagTrial >= self.nFrames()[1]:
            print("# We need to modify sync-frame-range end frame from %d to %d due to limited length of video 1." % \
                  (newSyncFrameRange[1], self.nFrames()[1]+lagTrial+1-1))
            newSyncFrameRange[1] = self.nFrames()[1]+lagTrial+1-1
        #
        print("# If lag is %d:" % (self.lagTrialRange[1]-1))
        lagTrial = self.lagTrialRange[1]-1
        print("#    Sync frames in video 0 are from %d to %d" % (newSyncFrameRange[0], newSyncFrameRange[1]-1))
        if newSyncFrameRange[0] < 0:
            print("# We need to modify sync-frame-range start from %d to 0." % newSyncFrameRange[0])
            newSyncFrameRange[0] = 0
        if newSyncFrameRange[1] > self.nFrames()[0]:
            print("# We need to modify sync-frame-range end frame from %d to %d due to limited length of video 0." % (newSyncFrameRange[1], self.nFrames()[0]))
            newSyncFrameRange[1] = self.nFrames()[0]
        print("#    Sync frames in video 1 are from %d to %d" % \
              (newSyncFrameRange[0] - lagTrial, \
               newSyncFrameRange[1]-1 - lagTrial))
        if newSyncFrameRange[0] - self.lagTrialRange[0] < 0:
            print("# We need to modify sync-frame-range start frame from %d to %d due to start frame of video 1." % \
                  (newSyncFrameRange[0], lagTrial))
            newSyncFrameRange[0] = lagTrial
        if newSyncFrameRange[1]-1 - lagTrial >= self.nFrames()[1]:
            print("# We need to modify sync-frame-range end frame from %d to %d due to limited length of video 1." % \
                  (newSyncFrameRange[1], self.nFrames()[1]+lagTrial+1-1))
            newSyncFrameRange[1] = self.nFrames()[1]+lagTrial+1-1
        # do the modification
        self.syncFrameRange = newSyncFrameRange.copy()    
    # end of def checkSyncFrameRange(self):
            
    def loadVideo(self, icam, printInfo=True):
        fVideo = os.path.join(self.workDir, self.fVideos[icam])
        fileExists = os.path.exists(fVideo)
        if printInfo:
            print('# Video %s existence is %s' % (fVideo, fileExists))
        if fileExists:
            self.videos[icam] = cv2.VideoCapture(fVideo)
            if printInfo:
                print('# The video can-be-opened is %s' %
                      self.videos[icam].isOpened())
            if self.videos[icam].isOpened():
                vidWidth = round(self.videos[icam].get(
                    cv2.CAP_PROP_FRAME_WIDTH))
                vidHeight = round(self.videos[icam].get(
                    cv2.CAP_PROP_FRAME_HEIGHT))
                self.videoResolutions[icam] = (vidWidth, vidHeight)
                self.videoFpss[icam] = self.videos[icam].get(cv2.CAP_PROP_FPS)
                self.videoFrameCounts[icam] = round(
                    self.videos[icam].get(cv2.CAP_PROP_FRAME_COUNT))
                if printInfo:
                    print("# Video resolution: ", self.videoResolutions[icam])
                    print("# Video frames per sec.: ", self.videoFpss[icam])
                    print("# Video frame count: ", self.videoFrameCounts[icam])
                    fourcc = int(self.videos[icam].get(cv2.CAP_PROP_FOURCC))
                    fourcc_str = chr(fourcc & 0xFF) + \
                        chr(fourcc >> 8 & 0x000000FF) + \
                        chr(fourcc >> 16 & 0x000000FF) + \
                        chr(fourcc >> 24 & 0x000000FF)
                    print("# Video fourcc: %s" % fourcc_str)
                # end of if printInfo:
            # end of if tryVid.isOpened():
        # end of if fileExists:
        if printInfo:
            print()
            
    def loadInitImg(self, icam):
        self.loadVideo(icam, printInfo=False)
        if type(self.videos[icam]) != cv2.VideoCapture:
            print("# Error. Cannot load video.")
            return
        print("# Loading template image from video file %d ..." % (icam+1), end='')
        for iframe in range(self.tmpltFrameIds[icam]+1):
            ret, frame = self.videos[icam].read()
        print("OK")
        self.imgInits[icam] = frame
        self.videos[icam].release()

    def pickTemplates(self, icam):
        """
        This function allows user to define the templates by mouse picking.
        1. If self.imgInits[icam] is empty or none, load the frame of image
           from the video.
        2. Popup the image and allow the user to pick the template by calling
           pickTemplates()
        3. Save the templates to self.tmplts, which is a 2D numpy array. 
           Each row (with 6 values) defines a template.
        Parameters
        ----------
        icam : TYPE int
            the camera (video) index you want to define templates. Index is
            0 based.
        Returns
        -------
        None.

        """
        # if imgInits[icam] is invalid, load the image
        if type(self.imgInits[icam]) != np.ndarray or \
           self.imgInits[icam].dtype != np.uint8 or \
           self.imgInits[icam].size <= 0:
            self.loadInitImg(icam)
            if type(self.videos[icam]) != cv2.VideoCapture:
                print("# Error. Cannot load video.")
                return
            # end of try (to read the template frame from video file)
        # set the file that templates are plotted to
        fTmpltFullpath = os.path.join(self.workDir, self.fTmplts[icam])
        fTmpltPlotFullpath = os.path.join(
            self.workDir, os.path.splitext(self.fTmplts[icam])[0] + '_templatePlots.JPG')
        self.tmplts[icam] = pickTemplates(
            img=self.imgInits[icam],
            savefile=fTmpltFullpath,
            saveImgfile=fTmpltPlotFullpath)
        pass
    # end of def pickTemplates(self, icam)
    
    def nFrames(self):
        return ss.videoFrameCounts
    
    def nPoints(self):
        return np.array( [self.tmplts[0].shape[0], self.tmplts[1].shape[0]], dtype=int)

    def defineBigTableKeys(self):
        self.bigTableKeys=[]
        self.bigTableKeys_1base=[]
        nP = max(self.nPoints()) # number of points (POI)
        nf = max(self.nFrames()) # number of frames (time steps)
        nc = 2 # number of cameras
        for icam in range(2):
            for ipoi in range(nP):
                # 0 base
                self.bigTableKeys.append(        't.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(       'xi.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(       'yi.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(     'roti.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(     'corr.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append('trackMode.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append('trackTime.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(    'resv1.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(    'resv2.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append(    'resv3.p%d.c%d'%(ipoi,icam))
                # 1 base
                self.bigTableKeys_1base.append(        't.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(       'xi.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(       'yi.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(     'roti.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(     'corr.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append('trackMode.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append('trackTime.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(    'resv1.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(    'resv2.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append(    'resv3.p%d.c%d'%(ipoi+1,icam+1))
            for iresv in range(10):
                # 0 base
                self.bigTableKeys.append('resv%d.c%d'%(iresv,icam))
                # 1 base
                self.bigTableKeys_1base.append('resv%d.c%d'%(iresv+1,icam+1))

        for ipoi in range(nP):
            # 0 base
            self.bigTableKeys.append('t.p%d'%(ipoi))
            self.bigTableKeys.append('xw.p%d'%(ipoi))
            self.bigTableKeys.append('yw.p%d'%(ipoi))
            self.bigTableKeys.append('zw.p%d'%(ipoi))
            for icam in range(nc):
                self.bigTableKeys.append('prjErrXi.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append('prjErrYi.p%d.c%d'%(ipoi,icam))   
            self.bigTableKeys.append('resvw1.p%d'%(ipoi))
            self.bigTableKeys.append('resvw2.p%d'%(ipoi))
            # 1 base
            self.bigTableKeys_1base.append('t.p%d'%(ipoi+1))
            self.bigTableKeys_1base.append('xw.p%d'%(ipoi+1))
            self.bigTableKeys_1base.append('yw.p%d'%(ipoi+1))
            self.bigTableKeys_1base.append('zw.p%d'%(ipoi+1))
            for icam in range(nc):
                self.bigTableKeys_1base.append('prjErrXi.p%d.c%d'%(ipoi+1,icam+1))
                self.bigTableKeys_1base.append('prjErrYi.p%d.c%d'%(ipoi+1,icam+1))   
            self.bigTableKeys_1base.append('resvw1.p%d'%(ipoi+1))
            self.bigTableKeys_1base.append('resvw2.p%d'%(ipoi+1))

    def allocateTables(self):
        nP = max(self.nPoints()) # number of points (POI)
        nf = max(self.nFrames()) # number of frames (time steps)
        nc = 2 # number of cameras
        # allocate memory
        self.bigTable = np.ones((nf, nc*nP*10+nc*10+nP*10),dtype=np.float64)*np.nan
        # create dictionary
        self.defineBigTableKeys()
        # # 0 base 
        # self.bDic={}
        # for ikey in range(len(self.bigTableKeys)):
        #     self.bDic[self.bigTableKeys[ikey]] = ikey
        # # 1 base
        # self.bDic_1base={}
        # for ikey in range(len(self.bigTableKeys_1base)):
        #     self.bDic_1base[self.bigTableKeys_1base[ikey]] = ikey+1            
        # for icam in range(2):
        #     for ipoi in range(nP):
        #         self.bDic[        't.p%d.c%d'%(ipoi+1,icam+1)]=0+10*ipoi+(10*nP+10)*icam
        #         self.bDic[       'xi.p%d.c%d'%(ipoi+1,icam+1)]=1+10*ipoi+(10*nP+10)*icam
        #         self.bDic[       'yi.p%d.c%d'%(ipoi+1,icam+1)]=2+10*ipoi+(10*nP+10)*icam
        #         self.bDic[     'roti.p%d.c%d'%(ipoi+1,icam+1)]=3+10*ipoi+(10*nP+10)*icam
        #         self.bDic[     'corr.p%d.c%d'%(ipoi+1,icam+1)]=4+10*ipoi+(10*nP+10)*icam
        #         self.bDic['trackMode.p%d.c%d'%(ipoi+1,icam+1)]=5+10*ipoi+(10*nP+10)*icam
        #         self.bDic['trackTime.p%d.c%d'%(ipoi+1,icam+1)]=6+10*ipoi+(10*nP+10)*icam
        #         self.bDic[    'resv1.p%d.c%d'%(ipoi+1,icam+1)]=7+10*ipoi+(10*nP+10)*icam
        #         self.bDic[    'resv2.p%d.c%d'%(ipoi+1,icam+1)]=8+10*ipoi+(10*nP+10)*icam
        #         self.bDic[    'resv3.p%d.c%d'%(ipoi+1,icam+1)]=9+10*ipoi+(10*nP+10)*icam
        #     for iresv in range(10):
        #         self.bDic['resv%d.c%d'%(iresv+1,icam+1)]  =iresv+10*nP  +(10*nP+10)*icam
        # for ipoi in range(nP):
        #     self.bDic['t.p%d'%(ipoi+1)]=0+ipoi*10+(10*nP+10)*nc
        #     self.bDic['xw.p%d'%(ipoi+1)]=1+ipoi*10+(10*nP+10)*nc
        #     self.bDic['yw.p%d'%(ipoi+1)]=2+ipoi*10+(10*nP+10)*nc
        #     self.bDic['zw.p%d'%(ipoi+1)]=3+ipoi*10+(10*nP+10)*nc
        #     for icam in range(nc):
        #         self.bDic['prjErrXi.p%d.c%d'%(ipoi+1,icam+1)]=0+2*icam+ipoi*10+(4+(10*nP+10)*nc)
        #         self.bDic['prjErrYi.p%d.c%d'%(ipoi+1,icam+1)]=1+2*icam+ipoi*10+(4+(10*nP+10)*nc)   
        #     self.bDic['resvw1.p%d'%(ipoi+1)]=4+2*nc+ipoi*10+(10*nP+10)*nc
        #     self.bDic['resvw2.p%d'%(ipoi+1)]=5+2*nc+ipoi*10+(10*nP+10)*nc
    # end of allocateTables(self):

    def saveBigTable(self, file='StereosyncV3_bigTable.csv'):
        try:
            header = self.bigTableKeys[:] # deep copy. 
        except:
            print("# Failed to get big table keys before saving to a file.")
        try:
            header[0] = '# ' + header[0]
            data_with_header = np.vstack((header, self.bigTable))
        except:
            print("# Failed to stack header to big table before saving to a file.")
        try:
            fBigTable = os.path.join(self.workDir, file)
            np.savetxt(fBigTable, data_with_header, delimiter=",", fmt="%s")
        except:
            print("# Failed to save the big table to %s." % fBigTable)
    # end of def saveBigTable(self, file='StereosyncV3_bigTable.csv'):

    def loadBigTable(self, file='StereosyncV3_bigTable.csv'):
        try:
            fBigTable = os.path.join(self.workDir, file)
            with open(fBigTable, "r") as file:
                header = file.readline().split(',')
        except:
            print("# Failed to load big table from file %s." % fBigTable)
        try:
            if header[0][0] == '#':
                header[0] = header[0].replace('#', '')
                header = [item.strip() for item in header]
                nheadercol = len(header)
                if nheadercol < 10:
                    print("# Warning: Header of big table file (%s) is too short. Check it." % fBigTable)
            else:
                self.defineBigTableKeys()
                print("# No header from big table file. It is defined separately.")
        except:
            print("# Error: Failed to read header line of the big table file %s" % fBigTable)
        try:
            bigTable = np.loadtxt(fBigTable, delimiter=',')
            ncol = bigTable.shape[1]
            if ncol != nheadercol:
                print("# Warning: Header has %d columns but tabular data has %d columns." % (ncol,))
            self.bigTable = bigTable[:] # use deep copy
            print("# Loaded big table from file. Shape:", self.bigTable.shape)
        except:
            print("# Error: Failed to load big table from file %s." % fBigTable)
    # end of def saveBigTable(self, file='StereosyncV3_bigTable.csv'):

    def eccTrackVideos(self, printInfo=True):
        for icam in self.trackCams:
            videoFilepath = self.videoFile(icam)
            tmplts = self.tmplts[icam]
            tmpltFrameId = self.tmpltFrameIds[icam]
            frameRange = self.trackFrameRange[icam]
            tmpltRange = self.trackPoints[icam]
            mTable = np.ones((max(frameRange+1), 6 * tmpltRange.size + 3), dtype=np.float64)*np.nan
            saveFilepath=None
            eccTrackVideo_v3(
                videoFilepath=videoFilepath,
                tmplts=tmplts,
                tmpltFrameId=tmpltFrameId,
                frameRange=frameRange,
                mTable=mTable,
    #            saveFilepath=saveFilepath
            )
            # copy mTable to self.bigTable
            for ipoi in tmpltRange:
#                start_col = self.bDic['xi.p%d.c%d'%(ipoi,icam)]
                start_col = self.bigTableKeys.index('xi.p%d.c%d'%(ipoi,icam))
                cols = np.arange(start_col, start_col+6)
                rows = np.arange(min(frameRange), max(frameRange))
                cols_mt = np.arange(ipoi*6, ipoi*6+6)
                self.bigTable[rows,cols[0]] = mTable[rows,cols_mt[0]].copy()
                self.bigTable[rows,cols[1]] = mTable[rows,cols_mt[1]].copy()
                self.bigTable[rows,cols[2]] = mTable[rows,cols_mt[2]].copy()
                self.bigTable[rows,cols[3]] = mTable[rows,cols_mt[3]].copy()
                self.bigTable[rows,cols[4]] = mTable[rows,cols_mt[4]].copy()
                self.bigTable[rows,cols[4]] = mTable[rows,cols_mt[5]].copy()
            # end of for ipoi in range(tmpltRange):
        # end of for icam in range(self.trackCams):
        print("# ")
        pass
    # end of def trackVideo(self):

    def findBestTimeLags(self, iPointsList=[], 
                         toPlotXi = False,
                         toPlotAllMse = False,
                         toPlotAllPrjErrs = False):
#        tLagsBestList = []
#        syncMsePrjErrList = []
        lagTrials = np.arange(self.lagTrialRange[0], self.lagTrialRange[1])
        # expand self.bestTimeLags if its length is too short (or zero)
        if self.bestTimeLags.shape[0] < min(self.nPoints()):
            sizeExpand = min(self.nPoints()) - self.bestTimeLags.shape[0]
            self.bestTimeLags = np.concatenate(
                (self.bestTimeLags, 
                 np.zeros(sizeExpand, dtype=self.bestTimeLags.dtype)))
        # re-allocate self.prjMseTable if its size is too small
        if self.prjMseTable.size < min(self.nPoints()) * lagTrials.size:
            if self.prjMseTable.shape[1] != lagTrials.size:
                self.prjMseTable = np.ones((min(self.nPoints()), lagTrials.size), dtype=float) * np.nan            
            else:
                sizeExpand = min(self.nPoints()) - self.prjMseTable.shape[0]
                self.prjMseTable = np.concatenate(
                    (self.prjMseTable, 
                     np.ones((sizeExpand, self.prjMseTable.shape[1]),dtype=self.prjMseTable.dtype) * np.nan))
        # sync by a certain point 
        if len(iPointsList) == 0:
            iPointsList = range(min(self.nPoints()))
        for ipoi in iPointsList:
            xi1_columns = []
            xi2_columns = []
#            xi1_columns.append(self.bDic['xi.p%d.c0' % (ipoi)])
#            xi1_columns.append(self.bDic['yi.p%d.c0' % (ipoi)])
#            xi2_columns.append(self.bDic['xi.p%d.c1' % (ipoi)])
#            xi2_columns.append(self.bDic['yi.p%d.c1' % (ipoi)])
            xi1_columns.append(self.bigTableKeys.index('xi.p%d.c0' % (ipoi)))
            xi1_columns.append(self.bigTableKeys.index('yi.p%d.c0' % (ipoi)))
            xi2_columns.append(self.bigTableKeys.index('xi.p%d.c1' % (ipoi)))
            xi2_columns.append(self.bigTableKeys.index('yi.p%d.c1' % (ipoi)))
            xi1 = self.bigTable[:, xi1_columns].copy()
            xi2 = self.bigTable[:, xi2_columns].copy()
            cmat1 = self.cams[0].cmat
            dvec1 = self.cams[0].dvec
            rvec1 = self.cams[0].rvec
            tvec1 = self.cams[0].tvec
            cmat2 = self.cams[1].cmat
            dvec2 = self.cams[1].dvec
            rvec2 = self.cams[1].rvec
            tvec2 = self.cams[1].tvec
            tLagsBest, syncMsePrjErr = syncByTriangulation(xi1, xi2, 
                                self.syncFrameRange,
                                lagTrials,
                                cmat1, dvec1, rvec1, tvec1, 
                                cmat2, dvec2, rvec2, tvec2,
                                toPlotXi=toPlotXi,
                                toPlotAllMse=toPlotAllMse,
                                toPlotAllPrjErrs=toPlotAllPrjErrs)
            self.bestTimeLags[ipoi] = tLagsBest
            self.prjMseTable[ipoi] = syncMsePrjErr.copy()
        # end of for ipoi in iPointsList:
        # define all time tags in bigTable, as self.bestTimeLags are determined
        icam = 0
        for ipoi in range(self.nPoints()[icam]):
#            col_idx = self.bDic['t.p%d.c%d' % (ipoi, icam)]
            col_idx = self.bigTableKeys.index('t.p%d.c%d' % (ipoi, icam))
            self.bigTable[0:self.nFrames()[icam], col_idx] = np.arange(0, self.nFrames()[icam])
        icam = 1
        for ipoi in range(self.nPoints()[icam]):
#            col_idx = self.bDic['t.p%d.c%d' % (ipoi, icam)]
            col_idx = self.bigTableKeys.index('t.p%d.c%d' % (ipoi, icam))
            self.bigTable[0:self.nFrames()[icam], col_idx] = np.arange(0, self.nFrames()[icam]) + self.bestTimeLags[ipoi]
    # end of def findBestTimeLags(...    
            
            
#            tLagsBestList.append(tLagsBest)
#            syncMsePrjErrList.append(syncMsePrjErr)
#        # end of for ipoi in range()
#        self.bestTimeLags = np.array(tLagsBestList, dtype=float)
#        self.prjMseTable = np.array(syncMsePrjErrList, dtype=float)

    def triangulatePoints(self):
        npoi = self.triangulationPoints.size
        cmat1 = self.cams[0].cmat
        dvec1 = self.cams[0].dvec
        rvec1 = self.cams[0].rvec
        tvec1 = self.cams[0].tvec
        cmat2 = self.cams[1].cmat
        dvec2 = self.cams[1].dvec
        rvec2 = self.cams[1].rvec
        tvec2 = self.cams[1].tvec
        for ipoi in range(npoi):
            poi = self.triangulationPoints[ipoi]
            xi1_columns = []
            xi2_columns = []
#            xi1_columns.append(self.bDic['xi.p%d.c0' % (poi)])
#            xi1_columns.append(self.bDic['yi.p%d.c0' % (poi)])
#            xi2_columns.append(self.bDic['xi.p%d.c1' % (poi)])
#            xi2_columns.append(self.bDic['yi.p%d.c1' % (poi)])
            xi1_columns.append(self.bigTableKeys.index('xi.p%d.c0' % (poi)))
            xi1_columns.append(self.bigTableKeys.index('yi.p%d.c0' % (poi)))
            xi2_columns.append(self.bigTableKeys.index('xi.p%d.c1' % (poi)))
            xi2_columns.append(self.bigTableKeys.index('yi.p%d.c1' % (poi)))
            xi1 = self.bigTable[:, xi1_columns].copy()
            xi2 = self.bigTable[:, xi2_columns].copy()
            # t1_triang is time tags for camera 1 to do triangulation, dtype is integer
            # t2_triang is time tags for camera 2 to do triangulation, dtype is "float"
            t1_triang_stt = self.triangulationFrameRange[0]
            t1_triang_end = self.triangulationFrameRange[1]
            if t1_triang_stt < 0:
                t1_triang_stt = 0
            if t1_triang_stt < self.trackFrameRange[0,0]:
                t1_triang_stt = self.trackFrameRange[0,0]
            if t1_triang_end > self.nFrames()[0]:
                t1_triang_end = self.nFrames()[0]
            if t1_triang_end > self.trackFrameRange[0,1]:
                t1_triang_end = self.trackFrameRange[0,1]
            t2_triang_stt = t1_triang_stt - self.bestTimeLags[ipoi]
            t2_triang_end = t1_triang_end - self.bestTimeLags[ipoi]
            if t2_triang_stt < 0:
                t1_triang_stt += int(abs(t2_triang_stt) + 1)
                t2_triang_stt = t1_triang_stt - self.bestTimeLags[ipoi]
            if t2_triang_stt < self.trackFrameRange[1,0]:
                t1_triang_stt += int(abs(self.trackFrameRange[1,0] - t2_triang_stt) + 1)
                t2_triang_stt = t1_triang_stt - self.bestTimeLags[ipoi]
            if t2_triang_end > self.nFrames()[1]:
                t1_triang_end -= int(abs(t2_triang_end - self.nFrames()[1]) + 1)
                t2_triang_end = t1_triang_end - self.bestTimeLags[ipoi]
            if t2_triang_end > self.trackFrameRange[1,1]:
                t1_triang_end -= int(abs(t2_triang_end - self.trackFrameRange[1,1]) + 1)
                t2_triang_end = t1_triang_end - self.bestTimeLags[ipoi]
            t1_triang = np.arange(t1_triang_stt, t1_triang_end, dtype=int)
            t2_triang = t1_triang - self.bestTimeLags[ipoi]
#            t1_triang = np.arange(
#                max(0, self.trackFrameRange[0][0], int(self.bestTimeLags[ipoi]+1), self.triangulationFrameRange[0]), 
#                min(self.nFrames()[0], self.trackFrameRange[0][1], int(self.trackFrameRange[1][1] + self.bestTimeLags[ipoi]),
#                    int(self.nFrames()[1] + self.bestTimeLags[ipoi]), self.triangulationFrameRange[1]),dtype=int)
            t2_triang = t1_triang - self.bestTimeLags[ipoi]
            nSteps = t1_triang.size
            xi1 = np.zeros((nSteps, 2), dtype=float)
            xi2 = np.zeros((nSteps, 2), dtype=float)
            # for camera 1, xi is a copy of sub-array of bigTable
            xi1[:,0] = self.bigTable[t1_triang, xi1_columns[0]].copy()
            xi1[:,1] = self.bigTable[t1_triang, xi1_columns[1]].copy()
            # for camera 2, xi has to be calculated by interpolation
            interp1_t = np.arange(self.bigTable.shape[0])
            interp1_xi2 = self.bigTable[:, xi2_columns[0]]
            interp1_yi2 = self.bigTable[:, xi2_columns[1]]
            interp1_xi2_obj = scipy.interpolate.interp1d(interp1_t, interp1_xi2, kind='cubic')
            interp1_yi2_obj = scipy.interpolate.interp1d(interp1_t, interp1_yi2, kind='cubic')
            xi2[:,0] = interp1_xi2_obj(t2_triang)
            xi2[:,1] = interp1_yi2_obj(t2_triang)
            # 
            objPoints, objPoints1, objPoints2,\
                prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
                triangulatePoints2(cmat1, dvec1, rvec1, tvec1, \
                                   cmat2, dvec2, rvec2, tvec2, \
                                   xi1, xi2)
            self.bigTable[t1_triang, self.bigTableKeys.index('xw.p%d'%ipoi)] = objPoints[:,0]
            self.bigTable[t1_triang, self.bigTableKeys.index('yw.p%d'%ipoi)] = objPoints[:,1]
            self.bigTable[t1_triang, self.bigTableKeys.index('zw.p%d'%ipoi)] = objPoints[:,2]
            self.bigTable[t1_triang, self.bigTableKeys.index('prjErrXi.p%d.c0'%ipoi)] = prjPoints1[:,0]
            self.bigTable[t1_triang, self.bigTableKeys.index('prjErrYi.p%d.c0'%ipoi)] = prjPoints1[:,1]
            self.bigTable[t1_triang, self.bigTableKeys.index('prjErrXi.p%d.c1'%ipoi)] = prjPoints2[:,0]
            self.bigTable[t1_triang, self.bigTableKeys.index('prjErrYi.p%d.c1'%ipoi)] = prjPoints2[:,1]
    # end of def triangulatePoints()
    
    def saveTimeLags(self, file='StereosyncV3_timeLags.csv'):
        try:
            fTimeLags = os.path.join(self.workDir, file)
            np.savetxt(fTimeLags, ss.bestTimeLags.reshape(-1,1), delimiter=",")
        except:
            print("# Error. StereosyncData.saveTimeLags(): Failed to save time lags to %s." % fTimeLags)
    # end of def saveTimeLags(self, file='StereosyncV3_bigTable.csv'):

    def loadTimeLags(self, file='StereosyncV3_timeLags.csv'):
        try:
            fTimeLags = os.path.join(self.workDir, file)
            ss.bestTimeLags = np.loadtxt(fTimeLags,delimiter=',')
        except:
            print("# Error. StereosyncData.loadTimeLags(): Failed to load time lags from %s." % fTimeLags)
    # end of def saveTimeLags(self, file='StereosyncV3_bigTable.csv'):

    def saveConfig(self, file='StereosyncV3_config.xml'):
        confFilename = os.path.join(workDir, file)
        # Create a FileStorage object for writing the XML file
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_WRITE)
        # Save config to file
        #   working directory
        fs.write('workDir', self.workDir)
        #   video files
        fs.write('fVideos_1', self.fVideos[0])
        fs.write('fVideos_2', self.fVideos[1])
        #   calibration files
        fs.write('fCalibs_1', self.fCalibs[0])
        fs.write('fCalibs_2', self.fCalibs[1])
        #   template frames
        fs.write('tmpltFrameIds', self.tmpltFrameIds)
        #   templates files
        fs.write('fTmplts_1', self.fTmplts[0])
        fs.write('fTmplts_2', self.fTmplts[1])
        #   range of tracking cameras
        fs.write('trackCams', self.trackCams)
        #   tracking points
        fs.write('trackPoints', self.trackPoints)
        #   range of tracking frames
        fs.write('trackFrameRange', self.trackFrameRange)
        #   range of possible lags
        fs.write('syncPoints', self.syncPoints)
        #   range of frames for sync analysis
        fs.write('syncFrameRange', self.syncFrameRange)
        #   range of points for triangulation
        fs.write('triangulationFrameRange', self.triangulationFrameRange)
        #   range of frames for triangulation
        fs.write('triangulationPoints', self.triangulationPoints)
        # release
        fs.release()
        print("# Config file saved.")
        pass
    # end of def saveConfig(self, file='StereosyncV3_config.xml'):

    def loadConfig(self, file='StereosyncV3_config.xml'):
        confFilename = os.path.join(workDir, file)
        # Create a FileStorage object for writing the XML file
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_READ)
        # loadfs config to file
        #   working directory
        self.workDir = fs.getNode('workDir').string()
        #   video files
        self.fVideos = ['','']
        self.fVideos[0] = fs.getNode('fVideos_1').string()
        self.fVideos[1] = fs.getNode('fVideos_2').string()
        #   calibration files
        self.fCalibs = ['','']
        self.fCalibs[0] = fs.getNode('fCalibs_1').string()
        self.fCalibs[1] = fs.getNode('fCalibs_1').string()
        #   template frames
        self.tmpltFrameIds = fs.getNode('tmpltFrameIds').mat().flatten()
        #   templates files
        self.fTmplts = ['','']
        self.fTmplts[0] = fs.getNode('fTmplts_1').string()
        self.fTmplts[1] = fs.getNode('fTmplts_1').string()
        #   range of tracking cameras
        self.trackCams = fs.getNode('trackCams').mat().flatten()
        #   tracking points
        self.trackPoints = fs.getNode('trackPoints').mat().flatten()
        #   range of tracking frames
        self.trackFrameRange = fs.getNode('trackFrameRange').mat().reshape(2,2)
        #   range of possible lags
        self.syncPoints = fs.getNode('syncPoints').mat().flatten()
        #   range of frames for sync analysis
        self.syncFrameRange = fs.getNode('syncFrameRange').mat().flatten()
        #   range of points for triangulation
        self.triangulationFrameRange = fs.getNode('triangulationFrameRange').mat().flatten()
        #   range of frames for triangulation
        self.triangulationPoints = fs.getNode('triangulationPoints').mat().flatten()
        fs.write('triangulationPoints', self.triangulationPoints)
        # release
        fs.release()
        print("# Config file loaded.")
        pass
    # end of def loadConfig(self, file='StereosyncV3_config.xml'):









if __name__ == '__main__':
    workDir = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10'
    fVideo1 = 'Cam 1.mp4'
    fVideo2 = 'Cam 2.mp4'
    fCalib1 = 'cam1_parameters.csv'
    fCalib2 = 'cam2_parameters.csv'
    fTmplt1 = 'templates1_test.csv'
    fTmplt2 = 'templates2_test.csv'

    ss = StereosyncData()
    ss.setWorkDir(workDir=workDir)
    ss.setVideoFile(icam=0, fVideo=fVideo1)
    ss.setVideoFile(icam=1, fVideo=fVideo2)
    ss.setCalibFile(icam=0, fCalib=fCalib1)
    ss.setCalibFile(icam=1, fCalib=fCalib2)
    ss.setTmpltFile(icam=0, fTmplt=fTmplt1)
    ss.setTmpltFile(icam=1, fTmplt=fTmplt2)

    ss.setTmpltFrameIds(icam=0, tmpltFrameId=0)
    ss.setTmpltFrameIds(icam=1, tmpltFrameId=0)
    ss.allocateTables()

    ss.setTrackCams([0, 1])
#    ss.trackCams = np.array([0,1])
    
    # set points that you want to track
    minNPoints = min(ss.nPoints())
    ss.setTrackPoints( np.arange(0, minNPoints))
#    ss.trackPoints = np.arange(0, min(ss.nPoints()))

    # set the start and end of frames to track (end is not included, Python style)
    nFrames1 = ss.nFrames()[0]
    nFrames2 = ss.nFrames()[1]
    ss.setTrackFrameRange(icam=0, theRange=(0, nFrames1))
    ss.setTrackFrameRange(icam=1, theRange=(0, nFrames1))   
#    ss.trackFrameRange[0]=np.array([0, ss.videoFrameCounts[0]])
#    ss.trackFrameRange[1]=np.array([0, ss.videoFrameCounts[1]])

    # for the first time, do tracking, which spends hours of time and save
    # for the later times, load the data from file
    toRunTracking = False
    if toRunTracking:    
        ss.eccTrackVideos()    
        ss.saveBigTable()
    else:
        ss.loadBigTable()
    
    # set synchronization settings
    ss.setLagTrialRange([-390, -375])
#    ss.lagTrialRange = np.array([-390,-375])
    
    # set points that are considered for synchronization
    ss.setSyncPoints(np.arange(0, 20))
#    ss.syncPoints = np.arange(0, 20)
    
    # set start and end frames that are considered for synchronization
    ss.setSyncFrameRange([360, 3960])
#    ss.syncFrameRange = np.array([360, 3960])
    
    # set triangulation data
    ss.setTriangulationPoints(np.arange(0, 20))
    ss.setTriangulationFrameRange([0, 5000])
#    ss.triangulationPoints = np.arange(0, 20)
#    ss.triangulationFrameRange = np.array([0, 5000])
    
    # sync
    import matplotlib.pyplot as plt
    ss.findBestTimeLags([0], toPlotXi=True, toPlotAllMse=True, toPlotAllPrjErrs=True)
    ss.findBestTimeLags()
    fig,ax=plt.subplots()
    ax.plot(ss.bestTimeLags); ax.grid(True)
    
    # do triangulation according to best time lags
    ss.triangulatePoints()

    # save
    # save all
#    ss.saveAll()
    ss.saveBigTable()
    ss.saveTimeLags()
    ss.saveConfig()
    
    
    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ss.bigTable[:,ss.bigTableKeys.index('t.p0.c0')],
             ss.bigTable[:,ss.bigTableKeys.index('xw.p0')]); ax.grid('on')
    plt.show()
    pass

    
    # 
    
    
    # tLagsBestList = []
    # syncMsePrjErrList = []
    # # sync by a certain point 
    # toPlotXi = False
    # toPlotAllMse = False
    # toPlotAllPrjErrs = False
    # for ipoi in range(20):
    #     xi1_columns = []
    #     xi2_columns = []
    #     xi1_columns.append(ss.bDic['xi.p%d.c0' % (ipoi)])
    #     xi1_columns.append(ss.bDic['yi.p%d.c0' % (ipoi)])
    #     xi2_columns.append(ss.bDic['xi.p%d.c1' % (ipoi)])
    #     xi2_columns.append(ss.bDic['yi.p%d.c1' % (ipoi)])
    #     xi1 = ss.bigTable[:, xi1_columns].copy()
    #     xi2 = ss.bigTable[:, xi2_columns].copy()
    #     cmat1 = ss.cams[0].cmat
    #     dvec1 = ss.cams[0].dvec
    #     rvec1 = ss.cams[0].rvec
    #     tvec1 = ss.cams[0].tvec
    #     cmat2 = ss.cams[1].cmat
    #     dvec2 = ss.cams[1].dvec
    #     rvec2 = ss.cams[1].rvec
    #     tvec2 = ss.cams[1].tvec
    #     tLagsBest, syncMsePrjErr = syncByTriangulation(xi1, xi2, 
    #                         ss.syncFrameRange,
    #                         ss.lagTrials,
    #                         cmat1, dvec1, rvec1, tvec1, 
    #                         cmat2, dvec2, rvec2, tvec2,
    #                         toPlotXi=toPlotXi,
    #                         toPlotAllMse=toPlotAllMse,
    #                         toPlotAllPrjErrs=toPlotAllPrjErrs)
    #     tLagsBestList.append(tLagsBest)
    #     syncMsePrjErrList.append(syncMsePrjErr)
    # # end of for ipoi in range()
    # tLagsBests = np.array(tLagsBestList, dtype=float)
    # syncMsePrjErrs = np.array(syncMsePrjErrList, dtype=float)
    
    
    
        
    
    
    
    
