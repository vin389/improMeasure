import numpy as np
import openpyxl as pxl
import os, sys

# a mss0p project file is a .xlsx file with a specific structure
# Worksheet "basic_info" 
#  The worksheet named "basic_info" contains basic information about the project.
#  In basic_info, the first column is the key and the second column is the value.
#  For example: num_cameras , 4 means the project has 4 cameras.
#  For example: num_pois , 10 means the project has 10 points of interest.
#  For example: num_steps , 100 means the project has 100 steps.
# Worksheet "camera_parameters"
#  The worksheet named "camera_parameters" contains camera parameters for each camera.
#  The first row is the name of the cameras (except the first column which is a constant "parameter_name").
#  The first column is the parameter name. From the 2nd row, they are image_width, image_height, 
#  rvec_x, rvec_y, rvec_z, 
#  tvec_x, tvec_y, tvec_z, cmat11_fx, cmat12, cmat13_cx, cmat21, cmat22_fy, cmat23_cy, cmat31, cmat32, cmat33,
#  k1, k2, k3, p1, p2, s1, s2, s3, s4, tau_x, tau_y 
#  
def load_project_file(project_file_path=None):
    # Four return values are initialized to None
    basic_info, pois_definition, camera_parameters, image_sources = None, None, None, None
    # open the project file 
    if project_file_path is None:
        # if project_file_path is not provided, pops up a tk file dialog to ask user to select the project file (an xlsx file)
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        filetypes = [("Excel files", "*.xlsx"), ("All files", "*.*")]
        project_file_path = filedialog.askopenfilename(
            title="Select a project file",
            filetypes=filetypes
        )
        # if user clicks cancel, this function returns None
        if not project_file_path:
            return None, None, None, None
    #
    if not os.path.isfile(project_file_path):
        raise FileNotFoundError(f"# Error: The project file {project_file_path} does not exist.")
    if not project_file_path.endswith('.xlsx'):
        raise ValueError(f"# Error: The project file {project_file_path} is not a valid .xlsx file.")
    # open the project file which is an xlsx file
    wb = pxl.load_workbook(project_file_path, data_only=True)
    # get the basic_info worksheet
    if 'basic_info' not in wb.sheetnames:
        raise ValueError("# Error: The project file does not contain a 'basic_info' worksheet.")
    ws_basic_info = wb['basic_info']
    # read the basic information from the basic_info worksheet
    basic_info = {}
    for row in ws_basic_info.iter_rows(min_row=2, max_col=2, values_only=True):
        key, value = row
        basic_info[key.strip()] = value.strip() if isinstance(value, str) else value
    pass

    return basic_info, pois_definition, camera_parameters, image_sources
