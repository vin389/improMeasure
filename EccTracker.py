import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from inputs import input2 
from pickTemplates import pickTemplates
from imshow2 import imshow2

# This class (EccTracker) is used to track a template in a video using the ECC 
# (Enhanced Correlation Coefficient) method.
# It allows user to assign the source of images (a video or a sequence of image), 
# set the templates to be tracked, and visualize the tracking process.

class EccTracker:
    def __init__(self):
        """
        Initialize the EccTracker class.
        """
        pass
        self.video_source = None
        self.video_source_type = None
#        self.templates = None
#        self.tracking_points = None
#        self.tracking_results = []
#        self.show_tracking = True
#        self.show_template = True
#        self.show_input = True
#        self.show_result = True

    def print_info(self):
        """
        Print the information of the EccTracker class.
        """
        print("EccTracker class")
        print("video_source:", self.video_source)
        print("templates:", self.templates)
        print("tracking_points:", self.tracking_points)
        print("tracking_results:", self.tracking_results)
        print("show_tracking:", self.show_tracking)
        print("show_template:", self.show_template)
        print("show_input:", self.show_input)
        print("show_result:", self.show_result)

    def set_video_source(self, video_path):
        """
        Set the source by giving the full path of a video file.
        If the file (video_path) does not exist, this function will raise a FileNotFoundError.
        """
        # check if the file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video source {video_path} does not exist.")
        # save the source
        self.video_source = video_path
        # set the type
        self.video_source_type = "video"

    def set_video_source_by_image_files(self, image_files):
        """
        Set the source by giving a list of image files.
        For example, ['/path/to/image1.jpg', '/path/to/image2.jpg', ...], 
        If any of the files does not exist, this function will raise a FileNotFoundError.
        """
        # check if the files exist
        for image_file in image_files:
            if not os.path.isfile(image_file):
                raise FileNotFoundError(f"Image file {image_file} does not exist.")
        # save the source
        self.video_source = image_files
        # set the type
        self.video_source_type = "image_files"

