import numpy as np
import scipy 
import cv2
import time
from multiprocessing import Process, Array

freq_vibration = 0.2
freq_predict = 100.0
t_start = time.time()
t_lag_measure = 0.5

# This is a fake measurement 
def fakeMeasure(n, t_array, x_array):  
  for i in range(n):
    # do a fake measurement 
    t_array[i] = time.time() - t_start
    xi = np.sin(2*np.pi*freq_vibration * t_array[i])
    # fake sleep (it is about the image measurement time cost)
    time.sleep(t_lag_measure)
    # write measurement to shared array 
    x_array[i] = xi

# This is a fake predictor
def predictor(n, t_array, x_array):  
    
  t_lastMeasured = -1
  
  while True:
    if t_lastMeasured + 1 >= n:
        break
    # create a black image and set draw settings
    img1 = np.zeros((500, 1500), dtype=np.uint8)
    img2 = np.zeros((500, 1500), dtype=np.uint8)
    
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 1
    text_pos = (50, 150)
    text_color = 128

    # Check
    if np.isnan(x_array[t_lastMeasured + 1]) == False:
        # if it is not nan, it means we have new measurement
        t_lastMeasured += 1 
        # plot info on the image 
        theStr = "# Real measure: Clock time:%.2f s. Found new real measure: %d: %.2f" \
              % (time.time() - t_start, \
                 t_lastMeasured, \
                 x_array[t_lastMeasured])
        cv2.putText(img1, theStr, text_pos, font_face, 
                    font_scale, text_color, thickness)
        cv2.imshow("Real measure", img1)
        cv2.waitKey(1)
       
                  
    # Predict
    t4 = np.nan
    if t_lastMeasured >= 2:
        t1 = t_array[t_lastMeasured - 2]
        t2 = t_array[t_lastMeasured - 1]
        t3 = t_array[t_lastMeasured - 0]
        x1 = x_array[t_lastMeasured - 2]
        x2 = x_array[t_lastMeasured - 1]
        x3 = x_array[t_lastMeasured - 0]
        t4 = time.time() - t_start
        coeff = np.polyfit(np.array([t1,t2,t3],dtype=float), 
                        np.array([x1,x2,x3],dtype=float), 
                        2)
        polynomial = np.poly1d(coeff)
        x4 = polynomial(t4)

    # wait
    time.sleep(1. / freq_predict)
    
    # Show/display/visualize/output
    #
    if np.isnan(t4) == False:
        text_pos2 = (50, 250)
        theStr2 = "# Predict. Clock time:%.2f s. Predict measure: (time: %.2f, x:%.2f)" \
              % (time.time() - t_start, \
                 t4, \
                 x4)
        cv2.putText(img2, theStr2, text_pos2, font_face,
                    font_scale, text_color, thickness)
    
    cv2.imshow('Predict', img2)
    cv2.waitKey(1)
#    time.sleep(1. / freq_check)



if __name__ == "__main__":
    
  # initialization, setting, ROI definition  
    
    
    
    
  # Number of terms in the Fibonacci series
  n = 120

  # Create a shared numpy array
  t_array = Array('f', n, lock=False)  # Avoid unnecessary locking overhead
  t_array[:] = np.ones(n, dtype=float) * np.nan
  x_array = Array('f', n, lock=False)  # Avoid unnecessary locking overhead
  x_array[:] = np.ones(n, dtype=float) * np.nan

  # Create processes
  p1 = Process(target=fakeMeasure, args=(n, t_array, x_array))
  p2 = Process(target=predictor, args=(n, t_array, x_array,))

  # Start processes
  p1.start()
  p2.start()

  # Wait for processes to finish
  p1.join()
  p2.join()

  print("Program completed!")
