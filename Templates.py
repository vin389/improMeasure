import numpy as np 
import cv2 as cv
import os, sys, time, math, copy

# The class Templates manages the templates for image tracking.
# It is used to load, save, and manipulate the templates. 
# Each template is represented by its position in image coordinates (xi and yi), the 
# upper left corner of the template (x0 and y0), and its width and height (width and height).
# There are 8 columns in the template data: name (string), id (int), xi, yi, x0, y0, width, and height.
# The name is a string, the id is an integer. 
# The x0, y0, width, and height are supposed to be integers, while xi and yi are floats.
# The class provides methods to load templates from a CSV file, save templates to a CSV file,
# and to create a template from a given image. 
# The class also provides methods to display the templates and to pick points on the templates. 
# The class is designed to be used with the OpenCV library for image processing. 
# The class functions include:
#    - __init__ (constructor): Given number of templates (n_templates) it initializes the template data 
#                   by creating a numpy array of shape (n_templates, 6) filled with NaN values.
#
#    - reset_demo(): 
#        Reset the template data to a demonstration data. 
#
#    - __repr__():
#        Returns a string representation of the template data.
#        The string is in CSV format and can be directly printed to a CSV file.
#
#    - num_templates(): 
#        Returns the number of templates.
#    
#    - resize(num_templates):
#        Resize the data by changing number of templates
#        If given num_templates is smaller, 
#
#    - get_image_points(points=None): 
#        Given a list of points, it returns the image points (n_given, 2).
#        If points is None, it returns all the image points (n_templates, 2).
#        For example, 
#          my_templates.get_image_points([0, 1, 2]) 
#          returns the image points for the 
#          first three. It is a numpy array of shape (:, 2). 
#
#    - get_templates(points=None): 
#        Given a list of points, it returns the templates (n_given, 6).
#        Un-defined templates will be filled with nan values. 
#        nan in any column of template data (xi, yi, x0, y0, w, h) means that the template is not defined yet.        
#        This function requires this file: pickTemplates.py
#        For example, 
#          my_templates.get_templates([0, 1, 2]) 
#          returns the templates for the
#          first three. It is a numpy array of shape (:, 6).
#
#      - get_template_images(points=None): 
#        Given a list of points and an image, it returns the template images.
#        For example, 
#          [tmplt_img_0, tmplt_img_1, tmplt_img_2] = my_templates.get_template_images([0, 1, 2], img) 
#        returns the template images for the first three. It is a list of images.
#        However, if the range of a template is out of the image, it shifts the range so that it is 
#        within the image. For example, if the template is out of the image by 2 pixels to the right edge
#        of the image, it shifts the range by 2 pixels to the left edge of the image.  
#
#      - draw_templates_on_image(points=None, img=None, color=(0, 255, 0), thickness=1): 
#        Given a list of points and an image, it draws the templates as rectangles on the image.
#        For example,
#          img_with_templates = my_templates.draw_templates_as_rectangles([0, 1, 2], img, color=(0, 255, 0), thickness=1)
#        returns the image with the templates drawn as rectangles.
#      
#      - pick_templates_from_image(points=None, img=None)
#        Given an image, it allows the user to pick points on the image and creates templates from them.
#        The given image can be a numpy array (height, width, channels) or (height, width), or a full-path file name.
#        If the image (img) is not given (default is None), it pops up a tk file dialog to select the image file.
#        The user can pick points by clicking on the image. The picked points are saved in the template data 
#        (i.e., self.templates_data).
#        For example, 
#          my_templates.pick_templates_from_image(points=[0, 3, 5], img=img)
#        allows the user to pick points on the image and set templates (by replacing) from them.
#        This function expands the template data if it is necessary. 
#        If templates_data is expanded, the new templates_data will be filled with nan values.
# 
#      - load_templates_from_csv:
#        Load templates from a CSV file.
#        The format is:
#        # Template data. The first line is the header and will be ignored.
#        # The data format is xi, yi, x0, y0, width, height

class Templates:
    # constructor. Given number of templates (n_templates) it initializes the template data.
    # The default value for n_templates is 0. If so, an empty array is created.
    # The arrray is self.templates_data, which is a numpy array of shape (n_templates, 6) filled with NaN values.
    def __init__(self, n_templates=0):
        if n_templates <= 0:
            n_templates = 0
        self.templates_data = np.ones((n_templates,6), dtype=np.float64) * np.nan
  
    def get_image_points(self, points=None):
        # Given a list of points, it returns the image points (n_given, 2).
        # For example, the_tmplts.get_image_points([0, 1, 2]) returns the image points for the 
        # first three. It is a numpy array of shape (:, 2).
        if points is None:
            points = range(self.templates_data.shape[0]) 
        n_given = len(points)
        templates_data = np.zeros((n_given,2),dtype=np.float64)
        for i in range(n_given):
            templates_data[i,0] = self.templates_data[points[i],0]
            templates_data[i,1] = self.templates_data[points[i],1]
        return templates_data
    
    def get_templates(self, points=None):
        # Given a list of points, it returns the templates (n_given, 6).
        # For example, the_tmplts.get_templates([0, 1, 2]) returns the templates for the
        # first three. It is a numpy array of shape (:, 6).
        if points is None:
            points = range(self.templates_data.shape[0])
        n_given = len(points)
        templates_data = np.zeros((n_given,6),dtype=np.float64)
        for i in range(n_given):
            templates_data[i,:] = self.templates_data[points[i],:]
        return templates_data
    
    def get_template_images(self, points=None, img=None):
        # Given a list of points and an image, it returns the template images.
        # For example, the_tmplts.get_template_images([0, 1, 2], img) returns the template images for the
        # first three. It is a list of images.
        # However, if the range of a template is out of the image, it shifts the range so that it is 
        # within the image. For example, if the template is out of the image by 2 pixels to the right edge
        # of the image, it shifts the range by 2 pixels to the left edge of the image.  
        if points is None:
            points = range(self.templates_data.shape[0])
        n_given = len(points)
        templates_data = np.zeros((n_given,6),dtype=np.float64)
        for i in range(n_given):
            templates_data[i,:] = self.templates_data[points[i],:]
        # get template images
        imgTmplts = []
        for i in range(n_given):
            x0 = int(templates_data[i,2])
            y0 = int(templates_data[i,3])
            w = int(templates_data[i,4])
            h = int(templates_data[i,5])
            xi = templates_data[i,0]
            yi = templates_data[i,1]
            # check if xi and yi are within the image
            if (xi < 0 or xi >= img.shape[1] or yi < 0 or yi >= img.shape[0]):
                print("# The point (%d,%d) is out of the image." % (xi,yi))
                continue
            # check if x0 and y0 are within the image
            if (x0 < 0 or x0 >= img.shape[1] or y0 < 0 or y0 >= img.shape[0]):
                print("# The template (%d,%d) is out of the image." % (x0,y0))
                continue
            # check if w and h are within the image
            if (w <= 0 or w >= img.shape[1] or h <= 0 or h >= img.shape[0]):
                print("# The template (%d,%d) has invalid size (%d,%d)." % (x0,y0,w,h))
                print("#   while the image size is (%d,%d)." % (img.shape[1],img.shape[0]))
                continue
            # check if the template is within the image
            # if the template is out of the image at the left edge, shift it to the right
            if (x0 < 0):
                x0 = 0
            if (y0 < 0):
                y0 = 0
            if (x0 + w >= img.shape[1]):
                x0 = img.shape[1] - w - 1
            if (y0 + h >= img.shape[0]):
                y0 = img.shape[0] - h - 1   
            # get the template image
            imgTmplts.append(img[y0:y0+h, x0:x0+w])
            # check if the template image is valid
            if (imgTmplts[i] is None or imgTmplts[i].size <= 0):
                print("# The template image (%d: %d,%d) is invalid." % (i+1, x0,y0))
                print("#    The template range is (%d,%d) to (%d,%d)." % (x0,y0,x0+w,y0+h))
                print("#    while the image size is (%d,%d)." % (img.shape[1],img.shape[0]))
                print("#    The template image is not valid.")
                continue
        # return the template images
        return imgTmplts
    
    def pick_templates_from_image(self, points=None, img=None):
#        Given an image, it allows the user to pick points on the image and creates templates from them.
#        The given image can be a numpy array (height, width, channels) or (height, width), or a full-path file name.
#        If the image (img) is not given (default is None), it pops up a tk file dialog to select the image file.
#        The user can pick points by clicking on the image. The picked points are saved in the template data 
#        (i.e., self.templates_data).
#        For example, 
#          my_templates.pick_templates_from_image(points=[0, 3, 5], img=img)
#        allows the user to pick points on the image and set templates (by replacing) from them.
#        This function expands the template data if it is necessary. 
#        If templates_data is expanded, the new templates_data will be filled with nan values.
        if points is None:
            points = range(self.templates_data.shape[0])
        n_given = len(points)
        # check img
        # if img is None, pop up a tk file dialog to select the image file and read it
        # if img is a string, it is supposed to be a file name
        # if img is a numpy array, it is supposed to be an image in opencv format 
        #    (height, width, channels) or (height, width)
        # all of the above cases will make the_img variable a copy of the image 

        # if img is None, pop up a tk file dialog to select the image file and read it 
        if img is None:
            # pop up a tk file dialog to select the image file 
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            # make sure the dialog window is on the top of the screen
            root.attributes("-topmost", True) 
            ufile = filedialog.askopenfilename(
                    title="Select an image file to pick templates", 
                    filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
            # destroy the dialog window
            root.destroy()
            # if user cancel the file dialog, exit the program
            if ufile == "":
                print("# User cancelled the file dialog.")
                return False
            # read image from the file
            the_img = cv.imread(ufile)
            # if the image is None or the image size is 0, exit the program
            if the_img is None or the_img.size <= 0:
                print("# Cannot read a valid image from file: ", img)
                return False
            # if the image is grayscale, convert it to BGR
            if len(the_img.shape) == 2:
                the_img = cv.cvtColor(the_img, cv.COLOR_GRAY2BGR)
        # if img is a string, it is supposed to be a file name
        if type(img) == str:
            the_img = cv.imread(img)
            # if the image is None or the image size is 0, exit the program
            if the_img is None or the_img.size <= 0:
                print("# Cannot read a valid image from file: ", img)
                return False
            # if the image is grayscale, convert it to BGR
            if len(the_img.shape) == 2:
                the_img = cv.cvtColor(the_img, cv.COLOR_GRAY2BGR)
        # check if the image is a numpy array or a file name
        # if img is a numpy array, make the_img variable a copy of img
        # if img is a file name, read the image from the file to the_img
        if isinstance(img, np.ndarray):
            if img.shape[0] <= 0 or img.shape[1] <= 0:
                print("# The image is empty.")
                return False
            the_img = copy.deepcopy(img)
            if len(the_img.shape) == 2:
                the_img = cv.cvtColor(the_img, cv.COLOR_GRAY2BGR)
        # pick templates from the image
        from pickTemplates import pickTemplates
        # if n_given is larger than current number of templates, expand the templates_data
        if n_given > self.templates_data.shape[0]:
            # expand the templates_data to the new size
            new_size = n_given
            self.templates_data = np.ones((new_size,6), dtype=np.float64) * np.nan
            # copy the old templates_data to the new templates_data
            self.templates_data[:self.num_templates(),:] = self.templates_data[:,:]
        # pick templates from the image (by using pickTemplates())
        new_templates = pickTemplates(the_img, nPoints=n_given, 
                                      nZoomIteration=0, # zoom until ESC is pressed 
                                      savefile='.', saveImgfile='.')
        # copy the new templates to the templates_data
        # according to the argument points. For example, if points=[0, 3, 5],
        # then the new templates will be copied to the 0th, 3rd, and 5th templates.
        for i in range(n_given):
            # check if the point is valid
            if points[i] < 0 or points[i] >= self.templates_data.shape[0]:
                print("# The point %d is out of the range." % points[i])
                continue
            # copy the new template to the templates_data
            self.templates_data[points[i],:] = new_templates[i,:]

    def draw_templates_on_image(self, points=None, img=None, color=(0, 255, 0), thickness=1):
#        Given a list of points and an image, it draws the templates as rectangles on the image.
#        For example,
#          img_with_templates = my_templates.draw_templates_as_rectangles([0, 1, 2], img, color=(0, 255, 0), thickness=1)
#        returns the image with the templates drawn as rectangles.
        if points is None:
            points = range(self.templates_data.shape[0])
        n_given = len(points)
        templates_data = np.zeros((n_given,6),dtype=np.float64)
        for i in range(n_given):
            templates_data[i,:] = self.templates_data[points[i],:]
        # if img is a string and is a file name, 
        # if img is None, pop up a tk file dialog to select the image file and read it
        if img is None:
            # pop up a tk file dialog to select the image file 
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            # make sure the dialog window is on the top of the screen
            root.attributes("-topmost", True) 
            ufile = filedialog.askopenfilename(
                    title="Select an image file to draw templates", 
                    filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
            # destroy the dialog window
            root.destroy()
            # if user cancel the file dialog, exit the program
            if ufile == "":
                print("# User cancelled the file dialog.")
                return False
            # read image from the file
            img = cv.imread(ufile)
        # draw rectangles
        for i in range(n_given):
            x0 = int(templates_data[i,2])
            y0 = int(templates_data[i,3])
            w = int(templates_data[i,4])
            h = int(templates_data[i,5])
            cv.rectangle(img, (x0,y0), (x0+w,y0+h), color=color, thickness=thickness)
        return img
    
    def load_templates_from_csv(self, filename=None):
        # Load templates from a CSV file.
        # The file should contain the templates in the following format:
        # xi, yi, x0, y0, width, height
        # The first line is the header and will be ignored.
        # if filename is "" or None, it pops up a tk file dialog to select a new file to read.
        # The file will be created if it does not exist.
        if filename == "" or filename is None:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            # make sure the dialog window is on the top of the screen
            root.attributes("-topmost", True)
            filename = filedialog.askopenfilename(
                title="Select a CSV file to load templates", 
                filetypes=[("CSV files", "*.csv")])
            # destroy the dialog window
            root.destroy()
            # if user cancel the file dialog, exit the program
            if filename == "":
                print("# User cancelled the file dialog.")
                return False
        # check if the file exists
        # if the file does not exist, exit the program
        if not os.path.isfile(filename):
            print("# File %s does not exist." % filename)
            return False
        try:
            self.templates_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
            print("# Templates loaded from file: %s" % filename)
            return True
        except Exception as e:
            print("# Error loading templates from file: %s" % filename)
            print("# Error: %s" % str(e))
            return False
        
    def save_templates_to_csv(self, filename=None):
        # Save templates to a CSV file.
        # The file will be created if it does not exist.
        # The format is:
        # xi, yi, x0, y0, width, height
        # if filename is "" or None, it pops up a tk file dialog to select a new file to write or overwrite.
        if filename == "" or filename is None:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            # make sure the dialog window is on the top of the screen
            root.attributes("-topmost", True)
            filename = filedialog.asksaveasfilename(
                title="Save templates to file", 
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")])
            # destroy the dialog window
            root.destroy()
            # if user cancel the file dialog, exit the program
            if filename == "":
                print("# User cancelled the file dialog.")
                return False
        # check if the file exists
        if os.path.isfile(filename):
            # ask user to overwrite the file
            print("# File %s already exists." % filename)
            print("# Do you want to overwrite it? (y/n)")
            answer = input("")
            if answer.lower() != "y":
                print("# User cancelled the file dialog.")
                return False
        # save the templates to the file
        try:
            np.savetxt(filename, self.templates_data, delimiter=',', header='# xi,yi,x0,y0,width,height', comments='')
            print("# Templates saved to file: %s" % filename)
            return True
        except Exception as e:
            print("# Error saving templates to file: %s" % filename)
            print("# Error: %s" % str(e))
            return False
        
    def unit_test_pickTemplates_full_image_from_file_dialog(self):
        # This function is a unit test for the pick_templates_from_image() function.
        # ask user to input number of templates from keyboard
        # pop up a tk input dialog that asks user to "Enter the number of templates"
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        # make sure the dialog window is on the top of the screen
        root.attributes("-topmost", True)
        # ask user to input number of templates
        # if user cancel the dialog, exit the program
        str_nTemplates = simpledialog.askinteger("Input", "Enter the number of templates (1-1000):",
                                             minvalue=1, maxvalue=1000)
        # destroy the dialog window
        root.destroy()
        # if user cancel the dialog, exit the program
        if str_nTemplates is None:
            print("# User cancelled the dialog.")
            return False
        try:
            nTemplates = int(str_nTemplates)
        except ValueError:
            print("# Error: The input is not a valid number.")
            return False
#        print("# Enter the number of templates:")
#        print("# For example, 3")
#        nTemplates = int(input(""))
        # check if the number of templates is valid
        if nTemplates <= 0:
            print("# Error: The number of templates is invalid.")
            return False
        # allocate the templates_data array
        self.templates_data = np.ones((nTemplates,6), dtype=np.float64) * np.nan
        # ask user to pick templates from image
        self.pick_templates_from_image(points=None, img=None)
        # print the templates data
        print("# Templates data:")
        print(self.templates_data)
        # ask user to save the templates data to a file
        self.save_templates_to_csv(filename=None)
        # print information
        print("# The templates data has been saved to the file you selected.")

    def unit_test_load_templates_from_csv(self):
        # This function is a unit test for the load_templates_from_csv() function.
        # ask user to input file name from keyboard
        # pop up a tk input dialog that asks user to "Enter the file name"
        self.load_templates_from_csv(filename=None)
        print(self.templates_data)
        pass


if __name__ == "__main__":
#    a = Templates(n_templates=3)
#    a.unit_test_pickTemplates_full_image_from_file_dialog()

    a = Templates()
    a.unit_test_load_templates_from_csv()

    pass
    pass
    pass



    
