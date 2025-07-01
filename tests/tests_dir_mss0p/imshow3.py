import cv2 
import numpy as np 
from drawTemplates3 import drawTemplates3


def imshow3(winname: str, img=np.ndarray, winmax: tuple=(1280, 720), interp:int=cv2.INTER_NEAREST_EXACT, 
            Xi=[], Xir=[], poi_names=[]):
    """
    This function is an enhanced imshow2 that not only allows user to pan and zoom
    the image by mouse dragging and wheel rolling, but also allows user to pick 
    points of interest (POIs) by mouse clicking and the template (or small regions
    of interest) of each POI. Each POI (with its template) has a name (a string).
    The POIs with templates are then saved to a file. 

    Pressing CTRL or ALT displays coordinates on the console screen and image
    itself, respectively.

    Key instructions ('q', 'ESC', 'x', 's', 'c', 'h', 't'):

    Pressing 'q', 'Q', or ESC key will exit the imshow window.
    
    Pressing 'x' or 'X' key will allow user to pick a point of interest (POI)
    (by using cv2.selectROI and the POI is at the center of the selectROI) 
    and its template (a small region of interest) (by using cv2.selectROI, and 
    the template is at the center of the selectROI). The POI name is asked
    through a tkinter input dialog for a string. 
    The POI is defined as a point (xi, yi) in the image, and is stored in a 
    n*2 np.float64 numpy array Xi, where n is the number of POIs.
    The template is defined as a rectangle with upper-left corner at (x0, y0)
    and its width and height are w and h, respectively. The template is stored
    in a n*4 np.int32 numpy array Tmplt, where n is the number of POIs.
    The return value is a tuple (Xi, Xir, poi_names), where Xi looks like this:
    Xi = np.array([[xi1, yi1], 
                   [xi2, yi2], ...], dtype=np.float64)
    and Tmplt looks like this:
    Xir = np.array([[x0_1, y0_1, w_1, h_1], 
                      [x0_2, y0_2, w_2, h_2], ...], dtype=np.int32)
    poi_names = ['poi_name_1', 'poi_name_2', ...]
    
    Pressing 's' or 'S' will save the POIs and their templates to a csv file.

    Pressing 'c' or 'C' will save the image with POIs and their templates to a file.

    Pressing 't' or 'T' will toggle the visibility of the templates on the image.
 
    Parameters
    ----------
    winname : str
        The name of the window to display the image.
    img : np.ndarray
        The image to display. It should be a BGR 3-channel or gray-scale 1-channel image.
    winmax : tuple, optional
        The maximum size of the window to display the image. It should be a tuple of two integers (width, height).
        The default is (1280, 720).
    interp : int, optional
        The interpolation method to use when resizing the image. It should be one of the OpenCV interpolation methods.
        It can be cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, or cv2.INTER_NEAREST_EXACT.
        See OpenCV documentation for more details about InterpolationFlags
        The default is cv2.INTER_NEAREST_EXACT.

    Returns
    -------
    (Xi, Xir, poi_names) : tuple
        A tuple containing:
        - Xi: a numpy array of shape (n, 2) with n being the number of POIs, 
              where each row is a point (xi, yi) in the image.
        - Xir: a numpy array of shape (n, 4) with n being the number of POIs,
                where each row is a rectangle defined by (x0, y0, w, h).
        - poi_names: a list of strings containing the names of the POIs.

    """
    # define x0, y0, x1, y1 as the region of image to display in imdhow window
    # x0, y0, x1, y1 are integers
    # x0 has to be between [0, x1)
    # x1 has to be between (x0, img.shape[1] - 1] 
    # y0 has to be between [0, y1)
    # y1 has to be between (y0, img.shape[0] - 1]
    
    # (x0, y0) is the upper-left pixel in the region to display    
    x0, x1, y0, y1 = 0, img.shape[1] - 1, 0, img.shape[0] - 1
    # x0_ is the previous status of x0. So are x1_, y0_, and y1_
    x0_, x1_, y0_, y1_ = -1, -1, -1, -1
    # mx is mouse cursor in window coord. mxi is that in image coord.
    mx, my, mxi, myi = 0, 0, 0, 0
    # mx_ is the previous status of mx
    mx_, my_, mxi_, myi_ = -1, -1, -1, -1
    # mbl, mbm, mbr are mouse-button status (left, middle, right buttons). 
    # 0 is released, 1 is being pressed
    mbl, mbm, mbr = 0, 0, 0
    # mbl_ is the previous status of mbl. So are mbm_ and mbr_ 
    mbl_, mbm_, mbr_ = -1, -1, -1    
    # scale is the current magnified scale of the image, which will be calculated later
    scale = 1.
    # wheel zoom factor
    wf = 1.2  
    max_scale = 30.0
    # about saved status right before dragging
    mx_before_dragging, my_before_dragging = 0, 0
    x0_before_dragging, y0_before_dragging = 0, 0
    x1_before_dragging, y1_before_dragging = 0, 0
    # mflags
    mflags, mflags_ = 0, 0
    
    def on_mouse(event, x, y, flags, params):
        nonlocal scale, winmax
        nonlocal mx, my, mxi, myi, x0, x1, y0, y1 
        nonlocal mbl, mbm, mbr
        nonlocal mx_, my_, mxi_, myi_, x0_, x1_, y0_, y1_ 
        nonlocal mbl_, mbm_, mbr_
        nonlocal mx_before_dragging, my_before_dragging
        nonlocal x0_before_dragging, y0_before_dragging
        nonlocal x1_before_dragging, y1_before_dragging
        nonlocal mflags, mflags_

#        print('\b'*100, end='')
#        print('x:%d y:%d     '%(x,y), end='')
        # update current status and previous status
        # mouse cursor location (window coordinate and image coordinate)
        mx_, my_, mxi_, myi_ = mx, my, mxi, myi
        mbl_, mbm_, mbr_ = mbl, mbm, mbr
        mx, my = x, y
        mxi = int(x0 + x / scale + 0.5)
        myi = int(y0 + y / scale + 0.5)
        mflags_ = mflags
        mflags = flags
        # update mbl, mbm, mbr, mbl_, mbm_, and mbr_
        if flags & cv2.EVENT_FLAG_LBUTTON != 0:
            mbl = 1
        else:
            mbl = 0
        if flags & cv2.EVENT_FLAG_MBUTTON != 0:
            mbm = 1
        else:
            mbm = 0
        if flags & cv2.EVENT_FLAG_RBUTTON != 0:
            mbr = 1
        else:
            mbr = 0
        # # if mouse left button is just clicked
        if mbl_ == 0 and mbl != 0:
            mx_before_dragging = mx
            my_before_dragging = my
            x0_before_dragging = x0
            y0_before_dragging = y0
            x1_before_dragging = x1
            y1_before_dragging = y1
        # if mouse left button is pressed and dragging
        if mbl_ == 1 and mbl == 1:
            dx = (-mx + mx_before_dragging) / scale
            dy = (-my + my_before_dragging) / scale
            x0 = int(x0_before_dragging + dx + 0.5)
            y0 = int(y0_before_dragging + dy + 0.5)
            x1 = int(x1_before_dragging + dx + 0.5)
            y1 = int(y1_before_dragging + dy + 0.5)
            # print("dragging: %d %d %d" % (mxi, mxi_before_dragging, dx))
            if x0 >= x1:
                x0 = x1 - 1
            if x0 < 0:
                x0_trial = 0
                x1_trial = x0_trial + (x1 - x0)
                x0 = x0_trial
                if x1_trial < img.shape[1]:
                    x1 = x1_trial
            if x1 >= img.shape[1]:
                x1_trial = (img.shape[1] - 1)
                x0_trial =  x1_trial - (x1 - x0)
                x1 = x1_trial
                if x0_trial >= 0:
                    x0 = x0_trial
            if y0 >= y1:
                y0 = y1 - 1
            if y0 < 0:
                y0_trial = 0
                y1_trial = y0_trial + (y1 - y0)
                y0 = y0_trial
                if y1_trial < img.shape[0]:
                    y1 = y1_trial                
            if y1 >= img.shape[0]:
                y1_trial = (img.shape[0] - 1)
                y0_trial =  y1_trial - (y1 - y0)
                y1 = y1_trial
                if y0_trial >= 0:
                    y0 = y0_trial
        if event == cv2.EVENT_MOUSEMOVE:
            mx, my = x, y
            mxi = x0 + x / scale 
            myi = y0 + y / scale
        if event == cv2.EVENT_MOUSEWHEEL:
            # mouse wheel up, zoom in 
            if flags > 0 and scale * wf < max_scale:
                mxi = x0 + x / scale 
                x0 = int(mxi - (mxi - x0) / wf + 0.5)
                x1 = int(mxi + (x1 - mxi) / wf + 0.5)
                myi = y0 + y / scale
                y0 = int(myi - (myi - y0) / wf + 0.5)
                y1 = int(myi + (y1 - myi) / wf + 0.5)
                if x0 >= x1:
                    x0 = x1 - 1
                if x0 < 0:
                    x0_trial = 0
                    x1_trial = x0_trial + (x1 - x0)
                    x0 = x0_trial
                    if x1_trial < img.shape[1]:
                        x1 = x1_trial
                if x1 >= img.shape[1]:
                    x1_trial = (img.shape[1] - 1)
                    x0_trial =  x1_trial - (x1 - x0)
                    x1 = x1_trial
                    if x0_trial >= 0:
                        x0 = x0_trial
                if y0 >= y1:
                    y0 = y1 - 1
                if y0 < 0:
                    y0_trial = 0
                    y1_trial = y0_trial + (y1 - y0)
                    y0 = y0_trial
                    if y1_trial < img.shape[0]:
                        y1 = y1_trial                
                if y1 >= img.shape[0]:
                    y1_trial = (img.shape[0] - 1)
                    y0_trial =  y1_trial - (y1 - y0)
                    y1 = y1_trial
                    if y0_trial >= 0:
                        y0 = y0_trial
            elif flags < 0:
                # mouse wheel down 
                mxi = x0 + x / scale 
                x0 = int(mxi - (mxi - x0) * wf + 0.5)
                x1 = int(mxi + (x1 - mxi) * wf + 0.5)
                myi = y0 + y / scale
                y0 = int(myi - (myi - y0) * wf + 0.5)
                y1 = int(myi + (y1 - myi) * wf + 0.5)
                if x0 >= x1:
                    x0 = x1 - 1
                if x0 < 0:
                    x0_trial = 0
                    x1_trial = x0_trial + (x1 - x0)
                    x0 = x0_trial
                    if x1_trial < img.shape[1]:
                        x1 = x1_trial
                if x1 >= img.shape[1]:
                    x1_trial = (img.shape[1] - 1)
                    x0_trial =  x1_trial - (x1 - x0)
                    x1 = x1_trial
                    if x0_trial >= 0:
                        x0 = x0_trial
                if y0 >= y1:
                    y0 = y1 - 1
                if y0 < 0:
                    y0_trial = 0
                    y1_trial = y0_trial + (y1 - y0)
                    y0 = y0_trial
                    if y1_trial < img.shape[0]:
                        y1 = y1_trial                
                if y1 >= img.shape[0]:
                    y1_trial = (img.shape[0] - 1)
                    y0_trial =  y1_trial - (y1 - y0)
                    y1 = y1_trial
                    if y0_trial >= 0:
                        y0 = y0_trial
        return

    # create image window    
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, on_mouse)
    # create empty lists for Xi, Xir, and poi_names
    # Xi and Xir will be converted to numpy arrays when returned
    if Xi is None or not isinstance(Xi, list):
        Xi = []  # list of points of interest (POIs) in the image
    if Xir is None or not isinstance(Xir, list):
        Xir = []  # list of templates (rectangles) for each POI
    if poi_names is None or not isinstance(poi_names, list):
        poi_names = []  # list of names for each POI
    # create a copy of the image to draw templates on it
    img_with_templates = img.copy()  # a copy of the image to draw templates on it
    img_with_templates = drawTemplates3(img, Xi, Xir, poi_names)
    show_img_with_templates = True
    # if img is not a 3-channel BGR image, convert it to a 3-channel BGR image``
    # the main look of imshow3
    while True:
        # update (zoom and resize) image
        if x0 != x0_ or y0 != y0_ or x1 != x1_ or y1 != y1_:
            # print("## %d %d %d %d" % (x0, x1, y0, y1))
            # generate new image
            scalex = winmax[0] / (x1 - x0)
            scaley = winmax[1] / (y1 - y0)
            scale = min(scalex, scaley)
            scaled_width  = int((x1 - x0) * scale + 0.5)
            scaled_height = int((y1 - y0) * scale + 0.5)
            if show_img_with_templates:
                imgScaled = cv2.resize(img_with_templates[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            else:
                imgScaled = cv2.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            # update 
            x0_, y0_, x1_, y1_ = x0, y0, x1, y1
        if (mxi != mxi_ or myi != myi_) and mflags & cv2.EVENT_FLAG_CTRLKEY != 0:
            print("\b"*100+"X:%.2f Y:%.2f Scale:%.2f " % (mxi, myi, scale), end='')            
        if (mxi != mxi_ or myi != myi_) and mflags & cv2.EVENT_FLAG_ALTKEY != 0:
            showStr = "X:%.2f Y:%.2f Scale:%.2f " % (mxi, myi, scale)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            color = (0, 255, 0)
            thickness = 2
            fontScale = 0.8
            if show_img_with_templates:
                imgScaled = cv2.resize(img_with_templates[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            else:
                imgScaled = cv2.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            cv2.putText(imgScaled, showStr, org, font, fontScale, color, thickness)
        cv2.imshow(winname, imgScaled)
        ikey = cv2.waitKey(30)
        # if "h" or "H" is pressed, pop up a help dialog (tkinter dialog)
        if ikey == ord("h") or ikey == ord("H"):
            help_string = "Help:\n  H or h: Pop this help dialog.\n"
            help_string += "  X or x: select a POI (at the crosshair),\n"
            help_string += "     then select an ROI for the POI (the box without crosshair),\n"
            help_string += "  T or t: taggle showing templates on the image.\n"
            help_string += "  C or c: save the image with POIs and their ROIs into a file.\n" 
            help_string += "  S or s: save the POIs and their ROIs data into a csv file.\n" 
            help_string += "  Q or q or ESC: quit.\n"
            # add poi and Xir to the help_string
            for i in range(len(poi_names)):
                help_string += "  %s: POI at (%.2f, %.2f) with template (%d, %d, %d, %d)\n" % (
                    poi_names[i], Xi[i][0], Xi[i][1], Xir[i][0], Xir[i][1], Xir[i][2], Xir[i][3])
            # popup the help_string to a tkinter dialog
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            # make this messagebox in front of every other windows
            root.attributes('-topmost', True)
            messagebox.showinfo("Help", help_string)
            # make focus back to the imshow window 
            # destroy the window first, then re-create it
            # to make sure the focus is on the imshow window
            cv2.destroyWindow(winname)
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, on_mouse)
            cv2.imshow(winname, imgScaled)
            ikey = cv2.waitKey(30)

        # if "q" or "Q" or ESC is pressed, quit
        if ikey == ord("q") or ikey == ord("Q") or ikey == 27:
            break
        # if "x" or "X" is pressed, allow user to select a poi (Xi) by selectROI (with crosshair)
        #                                  then select an roi for the poi (Xir) (without crosshair)
        if ikey == ord("x") or ikey == ord("X"):
            # pop up a tk information dialog shows the basic help message
            help_message = "You pressed T or t to select a Point of Interest (POI) and its template (POI region or Xir).\n"
            help_message += "Please select a POI by dragging a rectangle with crosshair, then select a template (Xir) by dragging another rectangle without crosshair.\n"
            help_message += "The POI name will be asked through a tkinter input dialog.\n"
            help_message += "Enter an empty name or cancel the dialog to skip this POI.\n"
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            root.attributes('-topmost', True)  # Make this dialog in front of every other windows
            messagebox.showinfo("Select POI and Xir", help_message)
            # ask user to pick a point of interest (POI) by selectROI
            roi = cv2.selectROI(winname, imgScaled, fromCenter=False, showCrosshair=True)
            if roi[2] <= 0 or roi[3] <= 0:
                print("# Warning: Invalid ROI selected. Please try again.")
                continue
            # roi is a tuple (x, y, w, h)
            scaled_x0_roi, scaled_y0_roi, scaled_w_roi, scaled_h_roi = roi
            x0_roi = x0 + scaled_x0_roi / scale 
            y0_roi = y0 + scaled_y0_roi / scale 
            w_roi = scaled_w_roi / scale 
            h_roi = scaled_h_roi / scale 
            xi = x0_roi + w_roi / 2 
            yi = y0_roi + h_roi / 2 
            print("# Selected POI at (%.2f, %.2f) with template size (%d, %d)" % (xi, yi, w_roi, h_roi))
            # ask user to select a template for the POI 
            roi = cv2.selectROI(winname, imgScaled, fromCenter=False, showCrosshair=False)
            scaled_x0_roi, scaled_y0_roi, scaled_w_roi, scaled_h_roi = roi
            x0_roi = int(x0 + scaled_x0_roi / scale + 0.5)
            y0_roi = int(y0 + scaled_y0_roi / scale + 0.5)
            w_roi = int(scaled_w_roi / scale) 
            h_roi = int(scaled_h_roi / scale)
            # ask user to input a name for the POI through a tkinter input dialog
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            poi_name = simpledialog.askstring("Input POI Name", "Enter a name for the POI that you just defined: (or empty name to skip this POI)")
            if poi_name is None or poi_name.strip() == "":
                print("# This POI is ignored.")
            else:
                # check if the poi_name exists. 
                if poi_name in poi_names:
                    # If it does, replace the old data with new data.
                    idx = poi_names.index(poi_name)
                    Xi[idx] = [xi, yi]
                    Xir[idx] = [x0_roi, y0_roi, w_roi, h_roi]
                else:
                    # If it does not exist, add the POI to the lists
                    Xi.append([xi, yi])
                    Xir.append([x0_roi, y0_roi, w_roi, h_roi])
                    poi_names.append(poi_name)
                # print the POI definition
                print("# POI definition: %s (%f, %f), (%d,%d,%d,%d)" % (poi_name, xi, yi, x0_roi, y0_roi, w_roi, h_roi))
            # draw the POI and its template on the image
            img_with_templates = drawTemplates3(img, Xi, Xir, poi_names)
            if show_img_with_templates:
                img_with_templates = drawTemplates3(img, Xi, Xir, poi_names)
                imgScaled = cv2.resize(img_with_templates[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            else:
                imgScaled = cv2.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            # make focus back to the imshow window
            cv2.destroyWindow(winname)
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, on_mouse)
            cv2.imshow(winname, imgScaled)
            ikey = cv2.waitKey(30)
        # end of if ikey == ord("x") or ikey == ord("X")
        if ikey == ord("t") or ikey == ord("T"):
            # toggle showing templates on the image
            show_img_with_templates = not show_img_with_templates
            if show_img_with_templates:
                img_with_templates = drawTemplates3(img, Xi, Xir, poi_names)
                imgScaled = cv2.resize(img_with_templates[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
            else:
                imgScaled = cv2.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                    dsize=(scaled_width, scaled_height), 
                                    interpolation=interp)
        # end of if ikey == ord("t") or ikey == ord("T")
        # if ikey == ord("s") or ikey == ord("S"):
        #  save the image with POIs and their ROIs into a file
        if ikey == ord("c") or ikey == ord("C"):
            # ask user to input a file name to save the image with POIs and their ROIs through file dialog
            import tkinter as tk
            from tkinter import filedialog
            root_filedialog = tk.Tk()
            root_filedialog.withdraw()  # Hide the root window
            root_filedialog.attributes('-topmost', True)
            save_path = filedialog.asksaveasfilename(
                title="Save Image with POIs and ROIs",
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("BMP files", "*.bmp")])
            if save_path:
                # save the image with POIs and their ROIs
                img_with_templates = drawTemplates3(img, Xi, Xir, poi_names)
                cv2.imwrite(save_path, img_with_templates)
                print("# Image with POIs and ROIs saved to %s" % save_path)
        # end of if ikey == ord("c") or ikey == ord("C") (save image file with templates)
        # if ikey == ord("s") or ikey == ord("S"):
        #  save the POIs and their ROIs data into a csv file. 
        if ikey == ord("s") or ikey == ord("S"):
            # ask user to input a csv file name to save the POIs and their ROIs through file dialog
            # The format is
            # poi_name, xi, yi, x0, y0, w, h
            # where (xi, yi) is the POI location, and (x0, y0, w, h) is the template rectangle
            if len(poi_names) == 0 or Xi.shape[0] == 0 or Xir.shape[0] == 0:
                print("# No POIs selected. Cannot save to file.")
                continue
            import os
            import tkinter as tk
            from tkinter import filedialog
            root_filedialog = tk.Tk()
            root_filedialog.withdraw() 
            root_filedialog.attributes('-topmost', True)
            save_path = filedialog.asksaveasfilename(
                title="Save POIs and ROIs Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")])
            if save_path:
                with open(save_path, 'w') as f:
                    f.write("poi_name,xi,yi,x0,y0,w,h\n")
                    for i in range(len(poi_names)):
                        f.write("%s,%.4f,%.4f,%d,%d,%d,%d\n" % (
                            poi_names[i], Xi[i][0], Xi[i][1], 
                            Xir[i][0], Xir[i][1], Xir[i][2], Xir[i][3]))
                print("# POIs and ROIs data saved to %s" % save_path)
        # end of if ikey == ord("s") or ikey == ord("S") (save data to csv file)

    try:
        cv2.destroyWindow(winname)
    except:
        pass
    return (np.array(Xi, dtype=np.float64),
            np.array(Xir, dtype=np.int32),
            poi_names)

def test_imshow3():
    image_path = r'c:/temp/example02.jpg'
    img = cv2.imread(image_path)
    imshow3("TEST IMSHOW3", img, interp=cv2.INTER_NEAREST_EXACT)


# if this script is run as a standalone script, run test_imshow3()
if __name__ == "__main__":
    winname = "test imshow3"
    winmax = (1280, 720)

    # use tkinter file dialog to ask user to select an image file (jpg, png, bmp, etc.)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # make file dialog on top of desktop
    root.attributes('-topmost', True)
    image_path = filedialog.askopenfilename(title="Select an Image File",
                                             filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not image_path:
        print("# No image file selected. Exiting.")
        exit()

    img = cv2.imread(image_path)
    if img is None:
        print("# Error: Could not read the image file. Exiting.")
        exit()

    # call imshow3 with the image and window name
    (poi, Xir, poi_names) = imshow3(winname, img, winmax=winmax, interp=cv2.INTER_NEAREST_EXACT)
    print("# imshow3 finished.")
    # if poi is not None and len(poi) >= 0 and poi is not None and poi is not empty, ... 
    #   we re-show the image with the POIs and templates
    if poi is None or len(poi) == 0 or Xir is None or len(Xir) == 0:
        poi_names = None
        print("# No POIs selected.")
    else:
        # convert poi and Xir to numpy arrays
        poi = np.array(poi, dtype=np.float64)
        Xir = np.array(Xir, dtype=np.int32)
        if len(poi) != len(Xir) or len(poi) != len(poi_names):
            print("# Error: POIs and templates do not match in length.")
            poi_names = None
        # print the result
        for i in range(len(poi_names)):
            print("# POI: %s at (%.2f, %.2f) with template (%d, %d, %d, %d)" % 
                (poi_names[i], poi[i][0], poi[i][1], Xir[i][0], Xir[i][1], Xir[i][2], Xir[i][3]))
        # draw it on images
        #from drawTemplates3 import drawTemplates3
        #img_with_templates = drawTemplates3(img, poi, Xir, poi_names)
        #imshow3("POIs with regions", img_with_templates, (800, 480), cv2.INTER_NEAREST_EXACT)



 

