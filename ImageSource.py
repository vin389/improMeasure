import time   # for time.sleep()
import os     # for os.path.exists()
import glob
import cv2
import numpy as np

class ImageSource:
    """
    A class to handle image acquisition from either a directory or webcam.
    """
    def __init__(self, src=None, start=None, count=None):
        """
        Initializes the ImageSource object by giving the path of files. 
        Usage 1: wildcard files
            theImgSrc = ImageSource(src='c:/dir/IMG_%04d.JPG', 
                              start=1234, count=10)
            the image source will be ['c:/dir/IMG_1234.JPG', 
                                  'c:/dir/IMG_1235.JPG', ...
                             ...
                                  'c:/dir/IMG_1243.JPG'] 
        Usage 2: glob files
            theImgSrc = ImageSource(src='c:/images/IMG_????.JPG')
            the image source will be all glob.glob(src) at the time 
            when constructor is called. 
        Usage 3: opencv camera 
            theImgSrc = ImageSource(src=0)
            the image source will be cv2.VideoCapture(src)
        
        """
        self.src_type = ''    # 'cstyle', wildcard', 'cvcam'
        self.cfiles = ''  # c-style file names (with %d, %nd, or %0nd)  
        self.cstart = -1  # c-style start index
        self.ccount = -1  # c-style file count
        self.wfiles = ''  # wildcard file names (with * or ??)
        self.cv_cam_idx = -1  # cvcam VideoCapture index as argument
        self.files = []       # list for full path of files (only for wildcard and glob)
        self.current_idx = 0  # current index of images (only for wildcard and glob)
    
        # usage 1: cfiles + start + count
        # Example: theImgSrc = ImageSource(src='c:/dir/IMG_%04d.JPG', 
        #                          start=1234, count=10)
        if (type(src) == str  
            and '%' in src 
            and type(start) == int
            and type(count) == int):
            self.set_cfiles(src, start, count)
    
    def set_cfiles(self, src, start, count):
        """
        Given src (e.g., "c:/img/IMG_%04d.JPG"),
              start (e.g., 10)
              count (e.g., 3)
        this function sets 
           self.cfiles as src (which has %), 
           self.cstart as start 
           self.ccount as count
           self.files as all files (full path)
        Returns
        -------
        None.
        """
        self.src_type = 'cstyle'
        self.cfiles = src
        self.cstart = start
        self.ccount = count
        self.files = []
        for i in range(self.cstart, (self.cstart + self.ccount)):
            self.files.append(self.cfiles % i)

    def num_files(self):
        """
        Returns number of files. 
        This function is only for self.src_type of "cstyle" and "wildcard"
        Returns
        -------
        TYPE int
            Number of files (only for self.src_type of "cstyle" and "wildcard".
        """
        if (self.src_type == 'cstyle'):
            return len(self.files)
        if (self.src_type == 'wildcard'):
            return len(self.files)
        else:
            return 0
  
    def get_file(self, index):
        if index >= self.num_files():
            return ''
        if index < -1:
            return ''
        if (self.src_type == 'cstyle'):
            return self.files[index]
        if (self.src_type == 'wildcard'):
            return self.files[index]
        
    def read(self):
        """
        Reads an image and return it.
        This function does not wait for file. If file does not exist, this 
        function prints a warning message and return None.
        Returns
        -------
            numpy.ndarray (opencv image)
            image that is read by cv.imread()
        """
        if self.src_type == 'cstyle' or self.src_type == 'wildcard':
            if self.current_idx < 0:
                return None
            if self.current_idx >= self.num_files():
                return None
            try:
                current_img = cv2.imread(self.get_file(self.current_idx))
                self.current_idx += 1
                return current_img
            except Exception:
                print("# Warning: ImageSource encountered error when running"
                      " cv2.imread(%s)" % self.get_file(self.current_idx))
                return None


    def read_with_wait(self, waitsec=0.1, warningsec=60):
        """
        Reads an image and return it.
        If the current file does not exist, this function waits [waitsec] 
        seconds. 
        If the current file does not exist, this function also prints a warning
        message every [warningsec] seconds. 
        Returns
        -------
            numpy.ndarray (opencv image)
            image that is read by cv.imread()
        """
        if self.src_type == 'cstyle' or self.src_type == 'wildcard':
            if self.current_idx < 0:
                return None
            if self.current_idx >= self.num_files():
                return None
            try:
                tic_before_wait = time.time()
                tic_last_warning = tic_before_wait
                # waiting for file
                while (True):
                    if os.path.exists(self.get_file(self.current_idx)):
                        break
                    time.sleep(waitsec)
                    tic_now = time.time()
                    if (tic_now - tic_last_warning) > warningsec:
                        print("# ImageSource: read_with_wait(): "
                              "Waiting for file %s "
                              "(have been waiting for %f seconds." 
                              % (self.get_file(self.current_idx),
                                 tic_now - tic_before_wait))
                        tic_last_warning = tic_now
                # file found     
                current_img = cv2.imread(self.get_file(self.current_idx))
                self.current_idx += 1
                return current_img
            except Exception:
                print("# Warning: ImageSource encountered error when running"
                      " cv2.imread(%s)" % self.get_file(self.current_idx))
                return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Returns
        -------
        the_str : str
            the string that display this object.
        """
        the_str = ''
        the_str += '# ImageSource\n'
        the_str += '#   src_type: %s\n' % self.src_type
        if self.src_type == 'cstyle':
            the_str += '#   cfiles: %s\n' % self.cfiles
            the_str += '#   cstart: %d\n' % self.cstart
            the_str += '#   ccount: %d\n' % self.ccount
            the_str += '#   files[0]: %s\n' % self.get_file(0)
            the_str += '#   files[-1]: %s\n' % self.get_file(-1)
            the_str += '#   current_idx: %d\n' % self.current_idx
            the_str += '#   current file: %s\n' % self.get_file(self.current_idx)
        return the_str

    def __del__(self):
        """
        Releases the capture object when the class is garbage collected.
        """
        pass
#        if self.cap is not None:
#            self.cap.release()



if __name__ == '__main__':
    # Example usage
    # Reading images from a directory
    imgsrc = ImageSource(r"D:\yuansen\ImPro\improMeasure\examples"
                         r"\ImageSource_read_with_wait\IMG_%04d.JPG",
        start=1, count=5)
    print(imgsrc)
    icount = 0
    while (True):
        print("# ImageSource is trying to read image %d: " % imgsrc.current_idx)
        img = imgsrc.read_with_wait(0.1, 5)
        if type(img) == np.ndarray and img.shape[0] > 0 and img.shape[1] > 0:
            icount += 1
        if type(img) == type(None) and icount == imgsrc.num_files():
            print("# ImageSource returns None. All images are read.")
 #   ret, image = image_source.read()
    
    # Reading from webcam (assuming webcam index 0)
 #   webcam_source = ImageSource(0)
 #   ret, frame = webcam_source.read()
    
    # Remember to release the capture object after use
 #   image_source.release()
 #   webcam_source.release()
    pass