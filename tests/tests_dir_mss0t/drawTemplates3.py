import numpy as np
import cv2

# drawTemplates3(image: np.ndarray, poi: np.ndarray = None, poir: np.ndarray = None, 
#                poi_names: list = None, draw_options: dict = None) -> np.ndarray:
# is a function that draws templates on an image.
# The templates are defined by poi (the image coordinates: n*2 np.float64 array), 
#                              poir (the roi of the poi: n*4 np.int32 array), 
#                              poi_names (a list of strings)
# 
# image: np.ndarray
#    The image on which the templates will be drawn. It should follow the OpenCV format, 
#    which is a NumPy array of shape (height, width, channels), where channels are usually 3 for RGB, 
#    or a single channel for grayscale images, which shape is (height, width).    
# 
# poi: np.ndarray
#    The image coordinates of each template. It is an n*2 np.float64 array, which looks like:
#    np.array([[xi_1, yi_1], 
#              [xi_2, yi_2], ...], dtype=np.float64)
#    The coordinates (xi, yi) does not need to be the exact center of the POI region (poir).
#
# poir: np.ndarray
#    The region of interest (ROI) for each template. It is an n*4 np.int32 array, which looks like:
#    np.array([[x1_1, y1_1, x2_1, y2_1],
#              [x1_2, y1_2, x2_2, y2_2], ...], dtype=np.int32)
#
# poi_names: list
#    A list of names (strings) for the templates. This is used to label the templates in the image.
#    It looks like:
#    ['template_name_1', 'template_name_2', ...]
#
# draw_options: Dict
#   draw_options['color'] is a list of 3 integers for RGB images, or a single integer for grayscale images.
#       The default color is [0, 255, 0] for green, or 255 for grayscale images.
#   draw_options['markerType'] is an OpenCV marker type. The options include:
#       cv2.MARKER_CROSS = 0 , cv2.MARKER_TILTED_CROSS = 1 , cv2.MARKER_STAR = 2 ,
#       cv2.MARKER_DIAMOND = 3 , cv2.MARKER_SQUARE = 4 , cv2.MARKER_TRIANGLE_UP = 5 ,
#       cv2.MARKER_TRIANGLE_DOWN = 6
#       The default marker type is cv2.MARKER_CROSS.
#   draw_options['markerSize'] is an integer for the size of the marker.
#       The default marker size is 9.
#   draw_options['thickness'] is an integer for the thickness of the marker.
#       The default thickness is 1.
#   draw_options['lineType'] is an integer for the line type. The options include:
#       cv2.LINE_4 = 4 , cv2.LINE_8 = 8 , cv2.LINE_AA = 16
#       The default line type is cv2.LINE_8.
#   draw_options['fontType'] is an OpenCV font type. The options include:
#       cv2.FONT_HERSHEY_SIMPLEX = 0 , cv2.FONT_HERSHEY_PLAIN = 1 ,
#       cv2.FONT_HERSHEY_DUPLEX = 2 , cv2.FONT_HERSHEY_COMPLEX = 3 ,
#       cv2.FONT_HERSHEY_TRIPLEX = 4 , cv2.FONT_HERSHEY_COMPLEX_SMALL = 5 ,
#       cv2.FONT_HERSHEY_SCRIPT_SIMPLEX = 6 , cv2.FONT_HERSHEY_SCRIPT_COMPLEX = 7
#       cv2.FONT_ITALIC = 16
#       The default font type is cv2.FONT_HERSHEY_SIMPLEX.
#   draw_options['fontColor'] is a list of 3 integers for RGB images, or a single integer for grayscale images.
#       The default font color is [0, 255, 0] for green, or 255 for grayscale images.
#   draw_options['fontScale'] is a float for the font scale of the text.
#       The default font scale is 1.0
#  
#   The return value is a copy of image which has the templates drawn on it. 
#   The input image is not modified. 
#
def drawTemplates3(image: np.ndarray, poi=None, poir=None, 
                   poi_names: list = None, draw_options: dict = None) -> np.ndarray:
# old version: 
#   def drawTemplates3(image: np.ndarray, templates: np.ndarray = None, 
#       template_names: list = None, draw_options: dict = {}) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        # print error with message starts with this function name
        print("# Warning: drawTemplates3: image must be a numpy.ndarray")
    if image.ndim not in [2, 3]:
        print("# Warning: drawTemplates3: image must be a 2D or 3D numpy array")
    if poi is None or poir is None or poi_names is None:
        return image
    if isinstance(poi, list):
        poi = np.array(poi, dtype=np.float64)
    if isinstance(poir, list):
        poir = np.array(poir, dtype=np.int32)
    if not isinstance(poi, np.ndarray) or not isinstance(poir, np.ndarray) or not isinstance(poi_names, list):
        return image
    if poi.shape[0] == 0 or poir.shape[0] == 0 or len(poi_names) == 0:
        return image
    if poi.shape[0] != poir.shape[0] or poi.shape[0] != len(poi_names):
        print("# Warning: drawTemplates3: poi, poir and poi_names must have the same number of elements")
        return image
    # Create a copy of the image to draw on
    image_clone = image.copy()

    # Set default drawing options if not provided
    if draw_options is None:
        draw_options = {}
    color = draw_options.get('color', [0, 255, 0])  # Default green for RGB, or 255 for grayscale
    markerType = draw_options.get('markerType', cv2.MARKER_CROSS)
    markerSize = draw_options.get('markerSize', 9)
    thickness = draw_options.get('thickness', 1)
    lineType = draw_options.get('lineType', cv2.LINE_8)
    fontType = draw_options.get('fontType', cv2.FONT_HERSHEY_SIMPLEX)
    fontColor = draw_options.get('fontColor', [0, 255, 0])  # Default green for RGB, or 255 for grayscale
    fontScale = draw_options.get('fontScale', 1.0)

    # Draw each template
    if not poi is None and poi.shape[0] > 0 and poi.shape[1] >= 2:
        # If poi is provided, use it to draw markers
        for i in range(poi.shape[0]):
            # get center point (integer based)
            x_center, y_center = poi[i,0], poi[i,1]
            center = (int(x_center+.5), int(y_center+.5))

            # Draw the marker at the center of the template
            cv2.drawMarker(image_clone, center, color=color, markerType=markerType,
                        markerSize=markerSize, thickness=thickness, line_type=lineType)
            # Draw the template names around at the center of the template
            cv2.putText(image_clone, poi_names[i], center, fontType,
                        fontScale, fontColor, thickness)

    if not poir is None and poir.shape[0] > 0 and poir.shape[1] >= 4:
        for i in range(poir.shape[0]):
        # If poir is provided, use it to draw rectangles
            # Draw the rectangle around the template
            top_left = (int(poir[i, 0]), int(poir[i, 1]))
            bottom_right = (int(poir[i, 0])+int(poir[i, 2]), int(poir[i,1])+int(poir[i, 3]))
            cv2.rectangle(image_clone, top_left, bottom_right, color=color, thickness=thickness)

    return image_clone


# Example usage:
#    A demonstration of how to use the drawTemplates3 function.
#    This program uses tkinter that pops up a file dialog to select an image file,
#    and creates 25 regularly distributed templates on the image.
#    The 25 regularly distributed are in a 5 by 5 pattern that are located
#    evenly and equally spaced in the image. 
#    This program displays the image with the templates drawn on it, 
#    and show it by using imshow2("Demo", image) function.

from imshow2 import imshow2

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image file
    image_path = filedialog.askopenfilename(title="Select an Image File",
                                             filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    
    if not image_path:
        print("No image file selected.")
        exit()

    # Read the image
    image = cv2.imread(image_path)

    # Create 25 regularly distributed templates in a 5 by 5 pattern
    height, width = image.shape[:2]
    poi = np.zeros((25, 2), dtype=np.float64)
    poir = np.zeros((25, 4), dtype=np.int32)
    poi_names = [f"Template {i+1}" for i in range(25)]

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            w = 20
            h = 20
            poi[idx, 0] = (j + 0.5) * width / 5  # x_center
            poi[idx, 1] = (i + 0.5) * height / 5  # y_center
            poir[idx, 0] = int(poi[idx, 0] - w / 2 +.5)
            poir[idx, 1] = int(poi[idx, 1] - h / 2 +.5)
            poir[idx, 2] = w
            poir[idx, 3] = h

    # Define drawing options
    draw_options = {
        'color': [0, 255, 0],         # Green color for RGB images
        'markerType': cv2.MARKER_CROSS,
        'markerSize': 9,
        'thickness': 1,
        'lineType': cv2.LINE_8,
        'fontType': cv2.FONT_HERSHEY_SIMPLEX,
        'fontColor': [0, 255, 0],     # Green color for RGB images
        'fontScale': 1.0
    }

    # Draw the templates on the image
    result_image = drawTemplates3(image, poi, poir, poi_names, draw_options)

    # Show the result image with drawn templates
    imshow2("Demo", result_image, winmax=(1200, 720))

