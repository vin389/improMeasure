import tkinter as tk
from tkinter import filedialog

def uiputfile(fileDialogTitle='Select the file to save', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    filePath = []
    fileName = []    
    tmpwin = tk.Tk()
    tmpwin.lift()    
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullname = filedialog.asksaveasfilename(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    if fullname:
        allIndices = [i for i, val in enumerate(fullname) if val == '/']
        filePath = fullname[0 : 1+max(allIndices)]
        fileName = fullname[1+max(allIndices) : ]
    return filePath, fileName
