import os
import json
import numpy as np
import cv2 as cv
from inputs import input2

#from r44FromCamposYawPitch import r44FromCamposYawPitch 
from pickPoints import pickPoints, pickPointAndTm
from pickTemplates import pickTemplates
from icf_r44FromCamposYawPitch import icf_r44FromCamposYawPitch
from icf_opticalFlow import icf_opticalFlow
from tkCalib import tkCalib
from icf_drawXyAnimation import icf_drawXyAnimation
from icf_drawFields import icf_drawFields
#from icf_wallMonitor_v2 import icf_wallMonitor_v2
from icf_wallMonitor_v3 import icf_wallMonitor_v3
from icf_wallMonitor_v4 import icf_wallMonitor_v4
from calcStrain3d import icf_calcQ4Strain3d

class improConsole2():
    # def load_config(self): 
    #    loads configuration from file
    #    --> self.config
    #      key: 'previousWorkDir': the working directory where 
    #           the previous time this program ran under.
    
    # Construction
    def __init__(self, debug=False):
        # variables for functions
        self.cnames = []    # command names (key-in command name)
        self.functions = [] # function calls (pointers)
        self.fhelps = []  # more details of help or information
        # misc variables
        self.debug = debug
        # load configuration from file 
        self.load_config()
        # load functions
        self.add_functions()
    
    def load_config(self):
        # Load configuration 
        # Loads configuration from file to class data
        # Class data: self.config
        #   'previousWorkDir': (str) the working directory
        # __file__ is the full path of current running file
        #          E.g., d:\\yuansen\\ImPro\\improMeasure\\improConsole2.py
        # configPath is the directory of the current running file
        # For example:
        #     configPath could be:
        #       "d:\\yuansen\\ImPro\\improMeasure"
        #     configFile could be 
        #       "d:\\yuansen\\ImPro\\improMeasure\\improConsole2_config.json"
        configPath = os.path.split(__file__)[0]
        configFile = os.path.join(configPath, 'improConsole2_config.json')
        # Read configuration --> config (a dict)
        #   If the configuration file is not there (see configFile), 
        #   this program creates one.
        try:
            with open(configFile, 'r') as f:
                self.config = json.load(f)
                print("# Loaded configuration from file. ")
        except:
            # 
            print('# No configuration file is found. Creating one ...', end='')
            # default config
            self.config = {'previousWorkDir': os.getcwd()}
            try:
                with open(configFile, 'w') as f:
                    json.dump(self.config, f)
                    print(' Done.') 
            except:
                print(' Cannot write a configuration file. Skipped.')
        # debug 
        if self.debug == True:
            self.print_config()
            
    def print_config(self):
        n_config_keys = len(list(self.config.keys()))
        print("# Configuration")
        for i in range(n_config_keys):
            config_key = list(self.config.keys())[i]
            config_value = self.config[config_key]
            print("#  ", config_key, ": ", config_value)

    def add_function(self, cname, function, fhelp):
        self.cnames.append(cname)
        self.functions.append(function)
        self.fhelps.append(fhelp)
        
    def run(self):
        while (True):
            print("# ===========================================")
            print("# Command    Description")
            print("# ...........................................")
            # print menu
            for i in range(len(self.cnames)):
                print('# %-10s %s' % (self.cnames[i], self.fhelps[i]))
            print("# ...........................................")
            print("# Type [command], or")
            print("# Type help [command], or")
            print("# Type q to quit:")
            # read user's input
            uInput = input2()
            uInput = uInput.strip()
            # check if user wants to quit
            if uInput == 'q' or uInput == 'quit' or uInput == 'return' or uInput == 'ret':
                print("# Bye.")
                break
            # check if user wants help
            # For example, if user inputs "help [command]" or 
            # "h [command]", this program runs help of the function 
            # of the command.
            if uInput[0:5] == 'help ' or uInput[0:2] == 'h ':
                cname_of_help = uInput.split()[1]
                for i in range(len(self.cnames)):
                    if cname_of_help == self.cnames[i]:
                        try:
                            help(self.functions[i])
                            break
                        except:
                            print("# This function does not support help.")
                            break                
            # run the function of the command
            for i in range(len(self.cnames)):
                if uInput == self.cnames[i]:
                    self.functions[i]()
                    break
                    
    def add_functions(self):
        self.add_function('tkCalib', tkCalib, 
            "GUI for single-image calibration.")
        self.add_function('wallMoni3', icf_wallMonitor_v3,
            "Runs wall crack and shear strain measurement")
        self.add_function('wallMoni4', icf_wallMonitor_v4,
            "Runs wall crack and shear strain measurement v4")
        self.add_function('drawFields', icf_drawFields,
            "Draws fields' colormap on an image")
        self.add_function('optflow', icf_opticalFlow,
            "Sparse optical flow on multiple images")
        self.add_function('strain3d', icf_calcQ4Strain3d,
            "Calculates surface strains given 3D Q4 coordinates before and after deformation")
        self.add_function('r44cyp', icf_r44FromCamposYawPitch, 
            "4x4 matrix from camera position, yaw, and pitch.")
        self.add_function('pps', pickPoints,
            "Pick N points from an image by mouse. Save csv file and image file.")
        self.add_function('ppt', pickTemplates,
            "Pick N points and templates from an image by mouse. Save csv file and image file.")
        self.add_function('xyanim', icf_drawXyAnimation,
            "Converts your x-y data to images that can generate a video")
#        self.add_function('wallMonitor2', icf_wallMonitor_v2,
#            "Runs wall crack and shear strain measurement")

    
if __name__ == '__main__':
    a = improConsole2(debug=True)
#    a.add_function('r44FromCamposYawPitch', r44FromCamposYawPitch, "No help")
    a.run()