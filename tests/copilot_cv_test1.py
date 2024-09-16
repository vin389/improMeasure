import cv2
import numpy as np

# This function creates an opencv window, allows the user to draw pen strokes on it, and returns the strokes as a mask.
# User can draw strokes by pressing and holding the left mouse button and moving the mouse. 
# The radius of the pen strokes can be adjusted by scrolling the mouse wheel.
# The strokes are returned as a binary mask, where the strokes are white and the background is black.
def drawStrokes(windowName='Draw Strokes', canvasSize=(512, 512), penRadius=10, penColor=(255, 255, 255)):
    # make penRadius a global variable
    # Create a blank canvas
    canvas = np.zeros((canvasSize[1], canvasSize[0], 3), np.uint8)
    # Create a window
    cv2.namedWindow(windowName)
    # Set the mouse callback function
    def onMouse(event, x, y, flags, param):
        penRadius = param['penRadius']
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(canvas, (x, y), penRadius, penColor, -1)
        # if the user scrolls the mouse wheel up, increase the pen stroke radius by 2
        # if the user scrolls the mouse wheel down, decrease the pen stroke radius by 2
        # the pen stroke radius is limited to be between 1 and 100
        # the pen stroke color is set to white
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                penRadius = min(100, penRadius + 2)
            else:
                penRadius = max(1, penRadius - 2)
            # set param['penRadius'] to the new penRadius
            param['penRadius'] = penRadius
            print("Set penRadius to %d" % penRadius)

    params = {'penRadius': penRadius}
    cv2.setMouseCallback(windowName, onMouse, param=params)
    # Loop until the user presses the 'q' key
    while True:
        cv2.imshow(windowName, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('['):
            penRadius = max(1, penRadius - 1)
        elif key == ord(']'):
            penRadius += 1
    # Destroy the window
    cv2.destroyWindow(windowName)
    # Convert the canvas to a binary mask
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask

# a main program to test the drawStrokes function
if __name__ == '__main__':
    mask = drawStrokes()
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # The drawStrokes function creates an opencv window, allows the user to draw pen strokes on it, and returns the strokes as a mask. User can draw strokes by pressing and holding the left mouse button and moving the mouse. The radius of the pen strokes can be adjusted by scrolling the mouse wheel. The strokes are returned as a binary mask, where the strokes are white and the background is black. The function can be used to create masks for image segmentation, annotation, or other purposes.
    # The drawStrokes function takes three optional arguments: windowName, canvasSize, and penRadius. The windowName argument specifies the name of the opencv window. The canvasSize argument specifies the size of the canvas in pixels. The penRadius argument specifies the radius of the pen strokes in pixels. The default values for these arguments are 'Draw Strokes', (512, 512), and 10, respectively.

