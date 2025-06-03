import numpy as np
import cv2 
# This class, PoiDefinition, is used to manage the definition of points of interest (POIs) in an image. 
# The POIs are typically defined by a photo that is the initial status of an experiment or an event. 
# It allows the user to manipulate the POIs, including:
# 1. Adding a new POI by mouse picking in the image
# 2. Removing existing POIs
# 3. Updating the coordinates of existing POIs
# 4. Saving the POIs to a file
# 5. Loading POIs from a file
# 6. Displaying (with zooming) the POIs on the image
# 7. Displaying the POIs in a table format
# The basic data used to store the POIs includes:
#   a numpy 
# 


class PoiDefinition:
    def __init__(self):
        pass

    def create_demo_data(self):
        # This function creates a demo data for the POI definition.
        # There are 8 synthetic points: (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
        # (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)
        # There are 4 synthetic cameras, located at ...
        from Camera import Camera
        n_cams = 4
        n_points = 8
        data = np.zeros((n_points, 3+n_cams*6), dtype=np.float64)
        # 8 points at (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)
        pos_3d = np.array([[1, 1, 1],[1, 1, -1],[1, -1, 1],[1, -1, -1], [-1, 1, 1],[-1, 1, -1],[-1, -1, 1],[-1, -1, -1]], dtype=np.float64)
        data[:, :3] = pos_3d
        # cam1  
        cam = Camera()
        cam.setRvecTvecByPosAim([3, 0, 0], [0, 0, 0])
        cam.setCmatByImgsizeFovs([1920, 1080], 90)
        cam.dvec = np.zeros((4, 1), dtype=np.float64)
        pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
        pos_2d = pos_2d_result[0].reshape(-1, 2)
        data[:, 3:5] = pos_2d
        tmplt_w, tmplt_h = 20, 20
        data[:, 5] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
        data[:, 6] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
        data[:, 7] = tmplt_w 
        data[:, 8] = tmplt_h 
        # cam2
        cam = Camera()
        cam.setRvecTvecByPosAim([0, 3, 0], [0, 0, 0])
        cam.setCmatByImgsizeFovs([1920, 1080], 90)
        cam.dvec = np.zeros((4, 1), dtype=np.float64)
        pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
        pos_2d = pos_2d_result[0].reshape(-1, 2)
        data[:, 9:11] = pos_2d
        tmplt_w, tmplt_h = 20, 20
        data[:, 11] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
        data[:, 12] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
        data[:, 13] = tmplt_w 
        data[:, 14] = tmplt_h 
        # cam3
        cam = Camera()
        cam.setRvecTvecByPosAim([-3, 0, 0], [0, 0, 0])
        cam.setCmatByImgsizeFovs([1920, 1080], 90)
        cam.dvec = np.zeros((4, 1), dtype=np.float64)
        pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
        pos_2d = pos_2d_result[0].reshape(-1, 2)
        data[:, 15:17] = pos_2d
        tmplt_w, tmplt_h = 20, 20
        data[:, 17] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
        data[:, 18] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
        data[:, 19] = tmplt_w 
        data[:, 20] = tmplt_h 
        # cam4
        cam = Camera()
        cam.setRvecTvecByPosAim([0, -3, 0], [0, 0, 0])
        cam.setCmatByImgsizeFovs([1920, 1080], 90)
        cam.dvec = np.zeros((4, 1), dtype=np.float64)
        pos_2d_result = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.cmat, cam.dvec)
        pos_2d = pos_2d_result[0].reshape(-1, 2)
        data[:, 21:23] = pos_2d
        tmplt_w, tmplt_h = 20, 20
        data[:, 23] = np.round(pos_2d[:,0] - tmplt_w / 2., 0) 
        data[:, 24] = np.round(pos_2d[:,1] - tmplt_h / 2., 0) 
        data[:, 25] = tmplt_w 
        data[:, 26] = tmplt_h 
        # copy to self.data
        self.data = data.copy()
        # name the POIs 
        self.poi_names = ['POI_%d' % (i+1) for i in range(n_points)]

    def num_of_pois(self):
        # This function returns the number of POIs.
        # The number of POIs is the number of rows in the data array.
        return self.data.shape[0]
    
    def num_of_cams(self):
        # This function returns the number of cameras.
        # The number of cameras is the number of columns in the data array minus 3 (x, y, z).
        return (self.data.shape[1] - 3) // 6

    def save(self, filename=None):
        # This function saves the POI data to a file.
        # If filename is None, pop up a tkinter file dialog to select the file.
        from tkinter import filedialog
        import tkinter as tk
        tmpwin = tk.Tk()
        tmpwin.lift()
        window.iconify()  # minimize to icon
        window.withdraw()  # hide it
        if filename is None:
            filename = filedialog.asksaveasfilename(title='Select the file to save', initialdir='/', filetypes=(('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), ))
        tmpwin.destroy()
        # save the data to the file
        # the first row is the header, which is:
        # "POI_name, xw, yw, zw, xi_1, yi_1, x0_1, y0_1, width_1, height_1, xi_2, yi_2, x0_2, y0_2, width_2, height_2, ..."
        # where the POI_name is the name of the POI, which is a string 
        # the 2nd, 3rd, and 4th columns are xw, yw, zw, which are the coordinates of the POI in the world coordinate system,
        # and are written in format of %24.16e
        # the 5th, 6th, columns are xi_1, yi_1, which are the coordinates of the POI in the image taken by the 1st camera,
        # the xi_1 and yi_1 are written in format of %24.16e
        # the 7th and 8th columns are x0_1 and y0_1, which are the integer coordinates of the left-top corner 
        # of the template in the image taken by the 1st camera, and are written in format of %d
        # the 9th and 10th columns are width_1 and height_1, which are the integer width and height of the template in 
        # the image taken by the 1st camera, and are written in format of %d
        # the rest of the columns are the same as above, but for the 2nd, 3rd, ... cameras.
        with open(filename, 'w') as f:
            # write the header
            header = "POI_name, xw, yw, zw"
            for i in range(self.num_of_cams()):
                header += ", xi_%d, yi_%d, x0_%d, y0_%d, width_%d, height_%d" % (i+1, i+1, i+1, i+1, i+1, i+1)
            f.write(header + "\n")
            # write the data
            for i in range(self.num_of_pois()):
                poi_name = self.poi_names[i]
                poi_data = self.data[i,:]
                poi_data_str = [poi_name] + ["%24.16e" % x for x in poi_data]
                f.write(",".join(poi_data_str) + "\n")
