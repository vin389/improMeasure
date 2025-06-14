import cv2 as cv
import numpy as np 

def imshow3(winname, img, winmax=(1280, 720), interp=cv.INTER_CUBIC):
    """
    This function is an enhanced imshow2 that not only allows user to pan and zoom
    the image by mouse dragging and wheel rolling, but also allows user to pick 
    points of interest (POIs) by mouse clicking and the template (or small regions
    of interest) of each POI. Each POI (with its template) has a name (a string).
    The POIs with templates are then saved to a file. 

    Pressing CTRL or ALT displays coordinates on the console screen and image
    itself, respectively.

    Pressing 'q', 'Q', or ESC key will exit the imshow window.
    Pressing 't' or 'T' key will allow user to pick a point of interest (POI)
    (by using cv2.selectROI and the POI is at the center of the selectROI) 
    and its template (a small region of interest) (by using cv2.selectROI, and 
    the template is at the center of the selectROI). The POI name is asked
    through a tkinter input dialog for a string. The POI and its template are
    stored in a dictionary pois_definition, which looks like this:
    pois_definition = {
        'poi_name_1': {'Xi':[xi, yi], 'Tmplt': [x0, y0, w, h]}, 
        ...
    }
    meaning that the POI named 'poi_name_1' is at pixel (xi, yi) in the image,
    and its template is a rectangle with upper-left corner at (x0, y0) and its
    width and height are w and h, respectively. The template is a small region. 

    Parameters
    ----------
    winname : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.
    winmax : TYPE, optional
        DESCRIPTION. The default is (1600, 900).
    interp : TYPE, optional
        DESCRIPTION. The default is cv.INTER_LINEAR.

    Returns
    -------
    None.

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
        if flags & cv.EVENT_FLAG_LBUTTON != 0:
            mbl = 1
        else:
            mbl = 0
        if flags & cv.EVENT_FLAG_MBUTTON != 0:
            mbm = 1
        else:
            mbm = 0
        if flags & cv.EVENT_FLAG_RBUTTON != 0:
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
        if event == cv.EVENT_MOUSEMOVE:
            mx, my = x, y
            mxi = x0 + x / scale 
            myi = y0 + y / scale
        if event == cv.EVENT_MOUSEWHEEL:
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
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, on_mouse)
    # create a dictionary for template storage
    pois_definition = {} 
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
            imgScaled = cv.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                  dsize=(scaled_width, scaled_height), 
                                  interpolation=interp)
            # update 
            x0_, y0_, x1_, y1_ = x0, y0, x1, y1
        if (mxi != mxi_ or myi != myi_) and mflags & cv.EVENT_FLAG_CTRLKEY != 0:
            print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bX:%.2f Y:%.2f Scale:%.2f " % (mxi, myi, scale), end='')            
        if (mxi != mxi_ or myi != myi_) and mflags & cv.EVENT_FLAG_ALTKEY != 0:
            showStr = "X:%.1f Y:%.1f Scale:%.1f " % (mxi, myi, scale)
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            color = (0, 255, 0)
            thickness = 2
            fontScale = 0.8
            imgScaled = cv.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                  dsize=(scaled_width, scaled_height), 
                                  interpolation=interp)
            cv.putText(imgScaled, showStr, org, font, fontScale, color, thickness)
        cv.imshow(winname, imgScaled)
        ikey = cv.waitKey(30)
        if ikey == ord("q") or ikey == ord("Q") or ikey == 27:
            break
        if ikey == ord("t") or ikey == ord("T"):
            # ask user to pick a point of interest (POI) by selectROI
            roi = cv.selectROI(winname, imgScaled, fromCenter=False, showCrosshair=True)
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
            roi = cv.selectROI(winname, imgScaled, fromCenter=False, showCrosshair=False)
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
            poi_name = simpledialog.askstring("Input POI Name", "Enter a name for the POI: (empty name to skip this POI)")
            if poi_name is None or poi_name.strip() == "":
                print("# This POI is ignored.")
            else:
                pois_definition[poi_name] = {}
                pois_definition[poi_name]['Xi'] = [xi, yi]
                pois_definition[poi_name]['Tmplt'] = [x0_roi, y0_roi, w_roi, h_roi]
            print("# POI definition: (%f, %f), (%d,%d,%d,%d)" % (pois_definition[poi_name]['Xi'][0],
                  pois_definition[poi_name]['Xi'][1],  pois_definition[poi_name]['Tmplt'][0],
                  pois_definition[poi_name]['Tmplt'][1], pois_definition[poi_name]['Tmplt'][2],
                  pois_definition[poi_name]['Tmplt'][3]))
            cv.destroyWindow(winname)
            cv.namedWindow(winname)
            cv.setMouseCallback(winname, on_mouse)

    try:
        cv.destroyWindow(winname)
    except:
        pass
    return pois_definition

def test_imshow3():
    image_path = r'c:/temp/example02.jpg'
    img = cv.imread(image_path)
    imshow3("TEST IMSHOW3", img, interp=cv.INTER_NEAREST)


# if this script is run as a standalone script, run test_imshow3()
if __name__ == "__main__":
    winname = "test imshow3"
    winmax = (1280, 720)
    img = cv.imread(r'c:/temp/example02.jpg')
    # if variable winname does not exist or it is not a string,
    # ask user to input a window name by keyboard
    if "winname" not in locals() or type(winname) != str:
        winname = input("# Input a window name (e.g., Test imshow3):")
    # if is not a string, or winname is empty, assign a default name
    if type(winname) != str or winname == "":
        winname = "TEST IMSHOW3"
    # if variable img is not defined, or it is not a numpy array, or it is empty
    # ask user to input an image file path through a tk file dialog
    if "img" not in locals() or type(img) != np.ndarray or img.size == 0:
        import tkinter as tk
        from tkinter import filedialog
        tk_root = tk.Tk()
        tk_root.withdraw()
        # make this file dialog in front of every other windows
        tk_root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename()
        img = cv.imread(file_path)    
        print("# Image file path: %s" % file_path)
        print("# Image shape: %s" % str(img.shape))
    # if variable winmax is not defined, or it is not a tuple of two integers, or it is empty
    # assign a default value
    # a valid winmax could be (1600, 900) or [1600, 900], or a numpy form of them.
    try: 
        winmax = np.array(winmax, dtype=int).flatten()
        winmax = tuple(winmax[0:2])
    except:
        # if winmax cannot be recognized, ask user to input a window size by entering two integers
        winmax = input("# Input window size (e.g., 1600 900):")
        winmax = winmax.split()
        winmax = tuple([int(winmax[0]), int(winmax[1])])

    
    # ... 






    winname = "TEST IMSHOW3"

#    test_imshow3()
    pois_definition = imshow3(winname, img, winmax=winmax, interp=cv.INTER_NEAREST)
    print("# imshow3 finished.")
    print(pois_definition)







 

