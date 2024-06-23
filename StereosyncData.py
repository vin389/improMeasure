import os
import sys
import time
import numpy as np
import cv2
import scipy

from inputs import input2
from pickTemplates import pickTemplates
from Camera import Camera

from eccTrackVideo import eccTrackVideo
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
                 trackPoints=[np.array([0,1,2], dtype=int), np.array([0,1,2], dtype=int)],
                 trackFrameRange=[np.array([0,3], dtype=int), np.array([0,3], dtype=int)],
                 lagTrials=np.array([-60,-30,0,30,60], dtype=int),
                 syncPoints=[np.array([0,1,2], dtype=int), np.array([0,1,2], dtype=int)],
                 syncFrameRange=np.array([120,4500], dtype=int),
                 triangulationPoints=np.array([0,1,2], dtype=int),
                 triangulationFrameRange=np.array([120,4500], dtype=int),
                 ):
        """
        This class manages the basic data for tkStereosync. 

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
            Full paths of camera parameters files. The default is ['',''].
              For example, ['cam1.csv', 'cam2.csv']
                Each file contains 21 to 31 scalars: width(1), height(1), 
                rvec(3), tvec(3), cmat(9), dvec(4 to 14)
        fTmplts : TYPE list of strings, optional
            Full paths of templates files. The default is ['',''].
              For example, ['tmplts1.csv', 'tmplts2.csv'].
                Each file contains N rows to define N templates. 
                Each template is defined by 6 scalars: xi, yi, x_upper_left,
                y_upper_left, width, height. The xi and yi are reals. The 
                latter 4 are integers. 
        tmpltFrameIds : TYPE list of integers, optional
            Frame indices of images that are used to define templates. 
            The default is [0,0].
            0 means that the template images are defined from the very first
            frame of the video. The templates do not need too be defined by 
            the first frame.
        trackCams : TYPE list of integers, optional
            Camera indices to run image point tracking. The default is [0, 1].
            0-based index. 0 is camera 1. 1 is camera 2.
        trackPoints : TYPE list of 1D integer arrays, optional
            trackPoints[0] is an np array, which contains points of camera 1 to track. 
            trackPoints[1] is an np array, which contains points of camera 2 to track. 
            Points are 0 based indices.
            The default is [ np.array([],dtype=int) , np.array([],dtype=int) ].
        trackFrameRange : TYPE list of 1D integer array, optional
            trackFrameRange[0] is an np array, which contains the starting frame ([0][0]) and end frame ([0][1]) of camera 1 to track. 
            trackFrameRange[1] is an np array, which contains the starting frame ([1][0]) and end frame ([1][1]) of camera 2 to track. 
            Frames are 0 based indices.
            The default is [[],[]].
        lagTrials : TYPE list of integers, optional
            time lags to try. The default is [].
            For example: [-60,-58,-56,...,56,58,60], has to be integers.
        syncPoints : TYPE list of 1D integer arrays, optional
            syncPoints[0] is an np array, which contains points that are used to 
              do the stereo synchronization. syncPoints[0] and[1] can be different 
              but have to be the same size.
            syncPoints[1] is an np array, which contains points that are used to 
              do the stereo synchronization. syncPoints[0] and[1] can be different 
              but have to be the same size.
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
        self.fVideos = fVideos[:] # deep copy everything
        self.fCalibs = fCalibs[:] # deep copy everything
        self.fTmplts = fTmplts[:] # deep copy everything. 
        self.tmpltFrameIds = tmpltFrameIds.copy() # deep copy everything. 
        self.trackCams = trackCams.copy()
        self.trackPoints = trackPoints[:] # deep copy everything
        self.trackFrameRange = trackFrameRange[:] # deep copy everything
        self.lagTrials = lagTrials.copy()
        self.syncPoints = syncPoints[:]
        self.syncFrameRange = syncFrameRange.copy()
        self.triangulationPoints = triangulationPoints[:]
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
        theStr.append("# tmpltFrameIds (indices are 0-base): %s" % (self.tmpltFrameIds).__str__())
        # display trackCams
        theStr.append("# trackCams (indices are 0-base): %s" % (self.trackCams).__str__())
        # display trackPoints
        theStr.append("# trackPoints 1 (indices are 0-base): %s" % (self.trackPoints[0]).__str__())
        theStr.append("# trackPoints 2 (indices are 0-base): %s" % (self.trackPoints[1]).__str__())
        # display trackFrameRange
        theStr.append("# trackFrameRange 1 (indices are 0-base, end is not included, Python style): %s" % self.trackFrameRange[0].__str__())
        theStr.append("# trackFrameRange 2 (indices are 0-base, end is not included, Python style): %s" % self.trackFrameRange[1].__str__())
        # display lagTrials
        theStr.append("# lagTrials: %s" % self.lagTrials.__str__())
        # display syncPoints
        theStr.append("# syncPoints 1 (indices are 0-base): %s" % (self.syncPoints[0]).__str__())
        theStr.append("# syncPoints 2 (indices are 0-base): %s" % (self.syncPoints[1]).__str__())
        # display syncFrameRange
        theStr.append("# syncFrameRange (indices are 0-base. end is not included, Python style): %s" % self.syncFrameRange[0].__str__())
                                   # the "-1" converts "python-style end (not included)" to "matlab-style end (included)". 
        # display triangulationPoints
        theStr.append("# triangulationPoints (indices are 0-base): %s" % (self.triangulationPoints).__str__())
        # display triangulationFrameRange
        theStr.append("# triangulationFrameRange (indices are 0-base. end is not included, Python style): %s" % self.triangulationFrameRange[0].__str()) 
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
                theStr.append("# imgInit %d: image not assigned yet." % (icam+1))
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
            self.bigTableKeys.append('tw.p%d'%(ipoi))
            self.bigTableKeys.append('xw.p%d'%(ipoi))
            self.bigTableKeys.append('yw.p%d'%(ipoi))
            self.bigTableKeys.append('zw.p%d'%(ipoi))
            for icam in range(nc):
                self.bigTableKeys.append('prjErrXi.p%d.c%d'%(ipoi,icam))
                self.bigTableKeys.append('prjErrYi.p%d.c%d'%(ipoi,icam))   
            self.bigTableKeys.append('resvw1.p%d'%(ipoi))
            self.bigTableKeys.append('resvw2.p%d'%(ipoi))
            # 1 base
            self.bigTableKeys_1base.append('tw.p%d'%(ipoi+1))
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
        # 0 base 
        self.bDic={}
        for ikey in range(len(self.bigTableKeys)):
            self.bDic[self.bigTableKeys[ikey]] = ikey
        # 1 base
        self.bDic_1base={}
        for ikey in range(len(self.bigTableKeys_1base)):
            self.bDic_1base[self.bigTableKeys_1base[ikey]] = ikey+1            
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
        #     self.bDic['tw.p%d'%(ipoi+1)]=0+ipoi*10+(10*nP+10)*nc
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
                start_col = self.bDic['xi.p%d.c%d'%(ipoi,icam)]
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
        tLagsBestList = []
        syncMsePrjErrList = []
        # expand self.bestTimeLags if its length is too short (or zero)
        if self.bestTimeLags.shape[0] < min(self.nPoints()):
            sizeExpand = min(self.nPoints()) - self.bestTimeLags.shape[0]
            self.bestTimeLags = np.concatenate(
                (self.bestTimeLags, 
                 np.zeros(sizeExpand, dtype=self.bestTimeLags.dtype)))
        # re-allocate self.prjMseTable if its size is too small
        if self.prjMseTable.size < min(self.nPoints()) * self.lagTrials.size:
            if self.prjMseTable.shape[1] != self.lagTrials.size:
                self.prjMseTable = np.ones((min(self.nPoints()), self.lagTrials.size), dtype=float) * np.nan            
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
            xi1_columns.append(self.bDic['xi.p%d.c0' % (ipoi)])
            xi1_columns.append(self.bDic['yi.p%d.c0' % (ipoi)])
            xi2_columns.append(self.bDic['xi.p%d.c1' % (ipoi)])
            xi2_columns.append(self.bDic['yi.p%d.c1' % (ipoi)])
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
                                self.lagTrials,
                                cmat1, dvec1, rvec1, tvec1, 
                                cmat2, dvec2, rvec2, tvec2,
                                toPlotXi=toPlotXi,
                                toPlotAllMse=toPlotAllMse,
                                toPlotAllPrjErrs=toPlotAllPrjErrs)
            self.bestTimeLags[ipoi] = tLagsBest
            self.prjMseTable[ipoi] = syncMsePrjErr.copy()
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
            xi1_columns.append(self.bDic['xi.p%d.c0' % (poi)])
            xi1_columns.append(self.bDic['yi.p%d.c0' % (poi)])
            xi2_columns.append(self.bDic['xi.p%d.c1' % (poi)])
            xi2_columns.append(self.bDic['yi.p%d.c1' % (poi)])
            xi1 = self.bigTable[:, xi1_columns].copy()
            xi2 = self.bigTable[:, xi2_columns].copy()
            # t1_triang is time tags for camera 1 to do triangulation, dtype is integer
            # t2_triang is time tags for camera 2 to do triangulation, dtype is "float"
            t1_triang = np.arange(self.triangulationFrameRange[0], self.triangulationFrameRange[1])
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
            self.bigTable[t1_triang, self.bDic['xw.p%d'%ipoi]] = objPoints[:,0]
            self.bigTable[t1_triang, self.bDic['yw.p%d'%ipoi]] = objPoints[:,1]
            self.bigTable[t1_triang, self.bDic['zw.p%d'%ipoi]] = objPoints[:,2]
            self.bigTable[t1_triang, self.bDic['prjErrXi.p%d.c0'%ipoi]] = prjPoints1[:,0]
            self.bigTable[t1_triang, self.bDic['prjErrYi.p%d.c0'%ipoi]] = prjPoints1[:,1]
            self.bigTable[t1_triang, self.bDic['prjErrXi.p%d.c1'%ipoi]] = prjPoints2[:,0]
            self.bigTable[t1_triang, self.bDic['prjErrYi.p%d.c1'%ipoi]] = prjPoints2[:,1]

        
#
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

    ss.trackCams = np.array([0,1])
    
    # set points that you want to track
    ss.trackPoints[0] = np.arange(0, ss.nPoints()[0])
    ss.trackPoints[1] = np.arange(0, ss.nPoints()[1])

    # set the start and end of frames to track (end is not included, Python style)
    ss.trackFrameRange[0]=np.array([0, ss.videoFrameCounts[0]])
    ss.trackFrameRange[1]=np.array([0, ss.videoFrameCounts[1]])

    # for the first time, do tracking, which spends hours of time and save
    # for the later times, load the data from file
    toRunTracking = False
    if toRunTracking:    
        ss.eccTrackVideos()    
        ss.saveBigTable()
    else:
        ss.loadBigTable()
    
    # set synchronization settings
    ss.lagTrials = np.arange(-480, -300+1)
    
    # set points that are considered for synchronization
    ss.syncPoints[0] = np.arange(0, 20)
    ss.syncPoints[1] = np.arange(0, 20)
    
    # set start and end frames that are considered for synchronization
    ss.syncFrameRange = np.array([360, 3960])
    
    # set triangulation data
    ss.triangulationPoints = np.arange(0, 20)
    ss.triangulationFrameRange = np.array([360, 3960])
    
    # sync
    import matplotlib.pyplot as plt
    ss.findBestTimeLags([0], toPlotXi=True, toPlotAllMse=True, toPlotAllPrjErrs=True)
    ss.findBestTimeLags()
    fig,ax=plt.subplots()
    ax.plot(ss.bestTimeLags); ax.grid(True)
    
    # do triangulation according to best time lags
    ss.triangulatePoints()
    
    iPointsList = range(2)
    cmat1 = ss.cams[0].cmat
    dvec1 = ss.cams[0].dvec
    rvec1 = ss.cams[0].rvec
    tvec1 = ss.cams[0].tvec
    cmat2 = ss.cams[1].cmat
    dvec2 = ss.cams[1].dvec
    rvec2 = ss.cams[1].rvec
    tvec2 = ss.cams[1].tvec
    for ipoi in iPointsList:
        xi1_columns = []
        xi2_columns = []
        xi1_columns.append(ss.bDic['xi.p%d.c0' % (ipoi)])
        xi1_columns.append(ss.bDic['yi.p%d.c0' % (ipoi)])
        xi2_columns.append(ss.bDic['xi.p%d.c1' % (ipoi)])
        xi2_columns.append(ss.bDic['yi.p%d.c1' % (ipoi)])
        xi1_all = ss.bigTable[:, xi1_columns].copy()
        xi2_all = ss.bigTable[:, xi2_columns].copy()
        t1_triang = np.arange(ss.triangulationFrameRange[0], ss.triangulationFrameRange[1])
        t2_triang = t1_triang         
        
        
    
    
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
    
    
    
        
    
    
    
    
