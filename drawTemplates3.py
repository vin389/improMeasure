import numpy as np
import cv2

# drawTemplates3(image: np.ndarray, templates: np.ndarray = None, template_names: list = None, draw_options: Dict = None) -> np.ndarray:
# is a function that draws templates on an image.
# The templates are defined a dictionary called pois_definition.
# 
# image: np.ndarray
#    The image on which the templates will be drawn. It should follow the OpenCV format, 
#    which is a NumPy array of shape (height, width, channels), where channels are usually 3 for RGB, 
#    or a single channel for grayscale images, which shape is (height, width).    
# 
# templates: np.ndarray
#    The array "templates" is a n by 6 array, where n is the number of templates.
#    Each row of the array represents a template that contains the following information:
#    templates[i, 0] = x-coordinate of the template center, which can be a float between two integers.
#    templates[i, 1] = y-coordinate of the template center, which can be a float between two integers.
#    templates[i, 2] = x-coordinate of the top-left corner of the template rectangle. It is an integer.
#    templates[i, 3] = y-coordinate of the top-left corner of the template rectangle. It is an integer.
#    templates[i, 4] = width of the template rectangle. It is an integer.
#    templates[i, 5] = height of the template rectangle. It is an integer.
#    For each template, a marker will be drawn at the center (Xi) of the template, and a 
#    rectangle will be drawn around the template defined by the coordinates (x, y, w, h).
#    The drawing options are specified in the 'draw_options' dictionary.
#
# template_names: list
#    A list of names (strings) for the templates. This is used to label the templates in the image. 
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
def drawTemplates3(image: np.ndarray, templates: np.ndarray = None, template_names: list = None, draw_options: dict = None) -> np.ndarray:
    # Create a copy of the image to draw on
    image_clone = image.copy()

    # Set default drawing options if not provided
    color = draw_options.get('color', [0, 255, 0])  # Default green for RGB, or 255 for grayscale
    markerType = draw_options.get('markerType', cv2.MARKER_CROSS)
    markerSize = draw_options.get('markerSize', 9)
    thickness = draw_options.get('thickness', 1)
    lineType = draw_options.get('lineType', cv2.LINE_8)
    fontType = draw_options.get('fontType', cv2.FONT_HERSHEY_SIMPLEX)
    fontColor = draw_options.get('fontColor', [0, 255, 0])  # Default green for RGB, or 255 for grayscale
    fontScale = draw_options.get('fontScale', 1.0)

    # Draw each template
    if not templates is None and templates.shape[0] > 0:
        if templates.shape[1] != 6:
            raise ValueError("Templates must have 6 columns: [x_center, y_center, x_top_left, y_top_left, width, height]")
        for i in range(templates.shape[0]):
            x_center, y_center, x_top_left, y_top_left, width, height = templates[i]
            center = (int(x_center+.5), int(y_center+.5))

            # Draw the marker at the center of the template
            cv2.drawMarker(image_clone, center, color=color, markerType=markerType,
                        markerSize=markerSize, thickness=thickness, line_type=lineType)

            # Draw the rectangle around the template
            top_left = (int(x_top_left), int(y_top_left))
            bottom_right = (int(x_top_left + width), int(y_top_left + height))
            cv2.rectangle(image_clone, top_left, bottom_right, color=color, thickness=thickness)

            # Draw the template names around at the center of the template
            cv2.putText(image_clone, template_names[i], center, fontType,
                        fontScale, fontColor, thickness)

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
    templates = np.zeros((25, 6), dtype=float)
    template_names = [f"Template {i+1}" for i in range(25)]

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            w = 20
            h = 20
            templates[idx, 0] = (j + 0.5) * width / 5  # x_center
            templates[idx, 1] = (i + 0.5) * height / 5  # y_center
            templates[idx, 2] = int(templates[idx, 0] - w / 2 +.5)
            templates[idx, 3] = int(templates[idx, 1] - h / 2 +.5)
            templates[idx, 4] = w
            templates[idx, 5] = h

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
    result_image = drawTemplates3(image, templates, template_names, draw_options)

    # Show the result image with drawn templates
    imshow2("Demo", result_image, winmax=(1200, 720))

