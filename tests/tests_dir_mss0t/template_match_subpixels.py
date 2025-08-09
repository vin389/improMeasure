"""
Function: template_match_subpixels
==================================

This function is designed to help you precisely track specific points
of interest (POIs) across a sequence of images. It's built on top of OpenCV's 'cv2.matchTemplate', but
it takes the accuracy a significant step further by estimating POI locations with *sub-pixel* precision.
This is achieved through a clever technique: parabolic fitting of the correlation map around the maximum
match point.

### What is it for? (Purpose)
This function is ideal for scientific and engineering applications where highly accurate displacement
tracking of small regions in image sequences is critical. Imagine you're analyzing microscopic movements,
tracking subtle deformations in materials, or monitoring precise shifts in an object's position –
'template_match_subpixels' is built for these scenarios. Each "template" you define consists of a
specific point of interest (the exact coordinate you want to track) and a bounding box region that
serves as the visual pattern to be matched.

### How is it different from 'cv2.matchTemplate'?
While 'cv2.matchTemplate' is a great starting point, 'template_match_subpixels' offers several key enhancements:
- **Sub-pixel Accuracy:** The most significant difference! Instead of just giving you integer pixel coordinates,
  this function refines the location to fractions of a pixel, providing much higher precision.
- **Multiple Template Support:** You're not limited to tracking just one thing. This function can handle
  multiple templates simultaneously, each with its own properties and even its own motion prediction.
- **Floating-Point Location Estimates:** It natively accepts and returns floating-point coordinates for POIs,
  which is essential for working with sub-pixel data.
- **Matching Quality Coefficient:** For every template, you'll receive a quantitative measure of how well it
  was matched in the current image. This "coefficient" helps you assess the reliability of each track.
- **Adaptive Search Range:** You can define a search area based on your expected displacement, which can
  make the tracking more efficient and robust, especially for larger movements.

### Input Parameters:
Let's break down what you need to provide to the function:

- 'init_img':
    - Type: Grayscale OpenCV image (NumPy array).
    - What it is: This is your "reference" or "initial" image. It's where you'll define the original
      locations of the points and templates you want to track.

- 'curr_img':
    - Type: Grayscale OpenCV image (NumPy array).
    - What it is: This is your "current" image, where the function will search for the templates
      defined from 'init_img'.

- 'init_templates':
    - Type: 'np x 6' NumPy array of 'float32'.
    - What it is: This is the core definition of your templates. Each row in this array represents
      one template and contains 6 crucial values: '[xi, yi, x0, y0, w, h]'.
        - '(xi, yi)': These are the *sub-pixel coordinates* of your Point of Interest (POI)
          within the 'init_img'. This is the exact point you want to track.
        - '(x0, y0)': These are the integer top-left corner (x, y) coordinates of the *bounding box*
          that defines the template region in 'init_img'. This region is the visual pattern that
          'cv2.matchTemplate' will use for matching.
        - '(w, h)': These are the integer width and height of the bounding box (template region).

- 'curr_poi' (Optional):
    - Type: 'np x 2' NumPy array of 'float32', or 'None'.
    - What it is: This parameter allows you to provide an *initial guess* for the POI locations
      '[xi, yi]' in the 'curr_img'.
    - Default: If you leave this as 'None' (which is often fine), the function will assume that
      your initial guess for the current POI locations is simply the '(xi, yi)' from 'init_templates'.
      This is a good starting point if you expect small movements.

- 'search_range_pixels' (Optional):
    - Type: a (np x 2) NumPy array of integers. For each template (each row), for example,  
        [srp_x_i, srp_y_i], it means, the search range is (srp_x_i) pixels both to the left and right, 
        and (srp_y_i) pixels both upward and downward. The entire search image size is conceptually
        (2 * srp_x_i + w) * (2 * srp_y_i + h)  
      If search_range_pixels is a two-integer list or tuple, a Numpy array of two integers, 
        say, [srp_x, srp_y], it will expand to 
        np.array([[srp_x, srp_y], [srp_x, srp_y], [srp_x, srp_y], ...] so that every template has the same
        search range factors. 
      If search_range_pixels is a single scalar, say, 'srp', it will expand to 
        np.ones((np_templates, 2), dtype=np.int32) * srp, so that every template has the same 
        search range pixels along both x and y. 
    - What it is: This controls how large the search area will be in 'curr_img' around your
      initial guess.
    - If you provide a single 'scalar' value (e.g., 30), and the template size is w * h, 
      the search image size will be (60+w) * (60+h) (i.e., (30 pixels to both the left and right, and w pixels of template itself))
    - If you provide (30, 5), and the template size is w * h
      the search image size will be (60+w) * (10+h) for all templates.
    - If you provide anything that can convert to a NumPy array of shape (np_templates, 2), you can specify different 
      search range pixels for every template.      

- 'search_range_factors' (Optional, only used if 'search_range_pixels' is None, empty, or not provided):
    - Type: a (np x 2) NumPy array of floats. For each template (each row), for example,  
        [srf_x_i, srf_y_i], it means, the search range is (w*srf_x_i) pixels both to the left and right, 
        and (h*srf_y_i) pixels both upward and downward. The entire search image size is conceptually
        (2 * w * srf_x_i + w) * (2 * h * srf_y_i + h)  
      If search_range_factors is a two-scalar list or tuple, a Numpy array of two floats, 
        say, [srf_x, srf_y], it will expand to 
        np.array([[srf_x, srf_y], [srf_x, srf_y], [srf_x, srf_y], ...] so that every template has the same
        search range factors. 
      If search_range_factor is a single float, say, 'srf', it will expand to 
        np.ones((np_templates, 2), dtype=np.float32) * srf, so that every template has the same 
        search range factor along both x and y. 
    - What it is: This controls how large the search area will be in 'curr_img' around your
      initial guess.
    - If you provide a single 'scalar' value (e.g., '2.5'), and the template size is w * h, 
      the search image size will be 6w * 6h (i.e., (2.5*w to both the left and right, and w pixels of template itself))
    - If you provide (2.5, 0.5), and the template size is w * h
      the search image size will be 6w * 2h for all templates.
    - If you provide anything that can convert to a NumPy array of shape (np_templates, 2), you can specify different 
      search range factors for every template.
    this factor will be applied to the
    

### Output Values:
The function will return two important pieces of information:

- 'updated_templates':
    - Type: 'np x 2' NumPy array of 'float32'.
    - What it contains: Each row '[xi_new, yi_new]' represents the *refined, sub-pixel accurate*
      location of the tracked POI in the 'curr_img'. This is your main result – the new position of your point!

- 'coeffs':
    - Type: 'np' sized NumPy array of 'float32'.
    - What it contains: Each value in this array is the *matching quality coefficient* for the
      corresponding template. A value closer to 1 indicates a very strong and confident match,
      while values closer to 0 or negative suggest a poor or no match. This helps you understand
      how well each point was tracked.

### Example Usage:
Let's see 'template_match_subpixels' in action with a few common scenarios.

'''python
import numpy as np
import cv2

# --- Create some dummy images for demonstration ---
# img1: A 300x300 white image with two black squares
img1 = np.ones((300, 300), dtype=np.uint8) * 255
cv2.rectangle(img1, (140, 140), (150, 150), (0, 0, 0), -1) # First square
cv2.rectangle(img1, (150, 150), (160, 160), (0, 0, 0), -1) # Second square

# img2: Same as img1, but squares are shifted slightly (e.g., by 1 pixel right, 2 pixels down)
img2 = np.ones((300, 300), dtype=np.uint8) * 255
cv2.rectangle(img2, (141, 142), (151, 152), (0, 0, 0), -1)
cv2.rectangle(img2, (151, 152), (161, 162), (0, 0, 0), -1)

# --- Case 1: Track a single template with default initial guess ---
print("--- Case 1: Tracking a single template ---")
# Define one template:
# POI at (150.0, 150.0)
# Template bounding box: top-left at (135, 135), width 30, height 30
init_templates_single = np.array([[150.0, 150.0, 135, 135, 30, 30]], dtype=np.float32)
new_positions_single, coeffs_single = template_match_subpixels(img1, img2, init_templates_single)
print("Updated POI (single):", new_positions_single)
print("Matching Coefficient (single):", coeffs_single)
print("-" * 40)

# --- Case 2: Track multiple templates ---
print("--- Case 2: Tracking multiple templates ---")
# Define two templates:
# Template 1: POI (150.0, 150.0), box (135, 135, 30, 30)
# Template 2: POI (160.0, 160.0), box (150, 150, 20, 20)
init_templates_multi = np.array([
    [150.0, 150.0, 135, 135, 30, 30],
    [160.0, 160.0, 150, 150, 20, 20]
], dtype=np.float32)
new_positions_multi, coeffs_multi = template_match_subpixels(img1, img2, init_templates_multi)
print("Updated POIs (multiple):", new_positions_multi)
print("Matching Coefficients (multiple):", coeffs_multi)
print("-" * 40)

# --- Case 3: Use a custom initial guess for the current POI location ---
print("--- Case 3: Using a custom initial guess ---")
# Let's say we expect the first point to have moved to (151.5, 152.5)
custom_curr_guess = np.array([[151.5, 152.5]], dtype=np.float32)
new_positions_custom_guess, coeffs_custom_guess = template_match_subpixels(
    img1, img2, init_templates_single, curr_pois=custom_curr_guess
)
print("Updated POI (custom guess):", new_positions_custom_guess)
print("Matching Coefficient (custom guess):", coeffs_custom_guess)
print("-" * 40)

# --- Case 4: Customize the search factor ---
print("--- Case 4: Customizing search factor ---")
# Increase search range by a factor of 2.0 for all templates
new_positions_large_search, coeffs_large_search = template_match_subpixels(
    img1, img2, init_templates_single, search_range_factors=2.0
)
print("Updated POI (large search):", new_positions_large_search)
print("Matching Coefficient (large search):", coeffs_large_search)
print("-" * 40)

# --- Case 5: Customize search factors for multiple templates individually ---
print("--- Case 5: Customizing search factors individually ---")
# Template 1: search factor 2.0, Template 2: search factor 1.5
individual_search_factors = np.array([2.0, 1.5], dtype=np.float32)
new_positions_individual_search, coeffs_individual_search = template_match_subpixels(
    img1, img2, init_templates_multi, search_range_factors=individual_search_factors
)
print("Updated POIs (individual search):", new_positions_individual_search)
print("Matching Coefficients (individual search):", coeffs_individual_search)
print("-" * 40)

# For visualization, you might want to draw the tracked points on curr_img
# (This part is for demonstration and requires a display environment)
# img_plot = img2.copy()
# for i, (xi, yi) in enumerate(new_positions_multi):
#     # Reconstruct the bounding box for plotting based on the new POI location
#     # and original template size/offset
#     x0_orig, y0_orig, w, h = init_templates_multi[i, 2:6].astype(int)
#     xi_orig, yi_orig = init_templates_multi[i, 0:2]
#     
#     # Calculate top-left of the bounding box relative to the new POI
#     box_x1 = int(xi - (xi_orig - x0_orig))
#     box_y1 = int(yi - (yi_orig - y0_orig))
#     
#     cv2.rectangle(img_plot, (box_x1, box_y1), (box_x1 + w, box_y1 + h), (0, 255, 0), 2)
#
# cv2.imshow('Tracked Points Visualization', img_plot)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''

"""

import numpy as np
import cv2

def template_match_subpixels(init_img, # initial image which POIs are defined from
                             curr_img, # current image where POIs are tracked
                             init_templates, # np x 6 array of float32, each row is [xi, yi, x0, y0, w, h]
                             curr_pois=None, # initial guess of np x 2 array of float32, each row is [xi, yi] for current POI locations
                             search_range_pixels=None, # np x 2 array of int32, each row is [srp_x_i, srp_y_i] for search range pixels for each template
                             search_range_factors=1.0): # np x 2 array of float32, each row is [srf_x_i, srf_y_i] for search range factors for each template
    # Ensure init_templates is a NumPy array of float32
    init_templates = np.array(init_templates, dtype=np.float32)
    np_templates = init_templates.shape[0] # Number of templates to track

    # If no initial guess for current template positions is provided,
    # assume they are the same as the initial POI locations.
    if curr_pois is None:
        curr_pois = init_templates[:, 0:2].copy()
    
    # Initialize an array to store matching coefficients for each template
    coeffs = np.zeros(np_templates)

    
    if search_range_pixels is None or search_range_pixels == 0 or len(search_range_pixels) == 0:
        # If search_range_pixels is not provided, use search_range_factors
        use_search_range_pixels = False

    if use_search_range_pixels:
        # Handle search_range_pixels:
        # try to convert search_range_pixels to a numpy array
        try:
            # try to convert search_range_pixels to a NumPy array
            search_range_pixels = np.array(search_range_pixels, dtype=np.int32).flatten()
        except Exception as e:
            use_search_range_pixels = False
        # if it has only one scalar
        if use_search_range_pixels and search_range_pixels.size <= 0:
            use_search_range_pixels = False
        if use_search_range_pixels and search_range_pixels.size == 1:
            search_range_pixels = np.ones((np_templates, 2), dtype=np.int32) * search_range_pixels[0]
        elif use_search_range_pixels and search_range_pixels.size == 2:
            # If it has two elements, expand it to a 2D array with the same number of pixels for all templates
            search_range_pixels = np.tile(search_range_pixels, (np_templates, 1)).astype(np.int32)
        elif use_search_range_pixels and search_range_pixels.size == np_templates * 2:
            # If it has the right size, reshape it to (np_templates, 2)
            search_range_pixels = search_range_pixels.reshape((np_templates, 2)).astype(np.int32)
        elif search_range_pixels.size > np_templates * 2:
            # If it has more than needed, take the first needed number of elements
            search_range_pixels = search_range_pixels[0:np_templates * 2]
            search_range_pixels = search_range_pixels.reshape((np_templates, 2)).astype(np.int32)
        else:
            # If it has an unexpected shape, take the first two elements and repeat them for all templates
            search_range_pixels = np.tile(search_range_pixels[0:2], (np_templates, 1)).astype(np.int32)

    if use_search_range_pixels == False:
        # Handle search_range_factors:
        # try to convert search_range_factors to a numpy array
        try:
            # try to convert search_range_factors to a NumPy array
            search_range_factors = np.array(search_range_factors, dtype=np.float32).flatten()
        except Exception as e:
            raise ValueError(f"Invalid search_range_factors: {search_range_factors}. "
                            "It should be a scalar, a 2-element list/tuple, or a NumPy array.")
            return None, None
        # if it has only one scalar
        if search_range_factors.size <= 0:
            raise ValueError("search_range_factors must be a positive scalar, a 2-element list/tuple, or a NumPy array.")
            return None, None
        if search_range_factors.size == 1:
            search_range_factors = np.ones((np_templates, 2), dtype=np.float32) * search_range_factors[0]
        elif search_range_factors.size == 2:
            # If it has two elements, expand it to a 2D array with the same factor for all templates
            search_range_factors = np.tile(search_range_factors, (np_templates, 1)).astype(np.float32)
        elif search_range_factors.size == np_templates * 2:
            # If it has the right size, reshape it to (np_templates, 2)
            search_range_factors = search_range_factors.reshape((np_templates, 2)).astype(np.float32)
        elif search_range_factors.size > np_templates * 2:
            # If it has more than needed, take the first needed number of elements
            search_range_factors = search_range_factors[0:np_templates * 2]
            search_range_factors = search_range_factors.reshape((np_templates, 2)).astype(np.float32)
        else:
            # If it has an unexpected shape, take the first two elements and repeat them for all templates
            search_range_factors = np.tile(search_range_factors[0:2], (np_templates, 1)).astype(np.float32)

    # Create a copy of curr_pois to store the updated (tracked) positions
    updated_templates = curr_pois.copy()

    # Iterate through each template to perform sub-pixel precision template matching
    for i in range(np_templates):
        # Extract template details from init_templates
        # xi, yi: sub-pixel POI in initial image
        # x0, y0, w, h: integer bounding box of the template in initial image
        xi, yi, x0, y0, w, h = init_templates[i]
        x0, y0, w, h = map(int, [x0, y0, w, h]) # Ensure bounding box coordinates are integers

        # Extract the template (the pattern to be matched) from the initial image
        template = init_img[y0:y0+h, x0:x0+w].copy()

        # Get the current guessed location of the POI in the current image
        gx, gy = curr_pois[i][0:2]

        # Calculate the offset of the POI relative to the template's top-left corner
        # This offset is crucial for translating the matched template position to the POI position
        dx = xi - x0
        dy = yi - y0

        # Get the dimensions of the current image
        img_h, img_w = curr_img.shape[:2]

        # Calculate the base search range dimensions.
        # This is based on the initial displacement guess and template size.
        search_range_base_x = 2 * abs(gx - xi) + w
        search_range_base_y = 2 * abs(gy - yi) + h
        
        # Apply the search_range_factor to determine the actual search area size
        search_range_x = int(search_range_base_x * search_range_factors[i,0])
        search_range_y = int(search_range_base_y * search_range_factors[i,1])

        # Define the coordinates for the search region in the current image.
        # Ensure the region stays within image boundaries (max/min with 0/img_dim).
        x1 = int(max(gx - search_range_x - dx, 0))
        y1 = int(max(gy - search_range_y - dy, 0))
        x2 = int(min(x1 + 2 * search_range_x + w, img_w))
        y2 = int(min(y1 + 2 * search_range_y + h, img_h))

        # Extract the search region from the current image
        search_region = curr_img[y1:y2, x1:x2].copy()

        # Check if the extracted search region is valid (large enough to contain the template)
        if search_region.shape[0] < h or search_region.shape[1] < w:
            # If not valid, set coefficient to 0 and skip to the next template
            coeffs[i] = 0
            continue

        # Perform standard template matching using OpenCV's TM_CCOEFF_NORMED method
        # This gives an initial integer-pixel best match location
        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res) # max_loc is (x_peak, y_peak) in 'res'

        # Initialize sub-pixel refinements and coefficients
        x_peak, y_peak = max_loc[0], max_loc[1]
        dx_sub = 0.0
        dy_sub = 0.0
        coeff_x = max_val # Initial correlation value at the integer peak
        coeff_y = max_val # Initial correlation value at the integer peak

        # Define a kernel for parabolic fitting. This kernel helps determine the coefficients
        # of the parabola based on three points (peak and its neighbors).
        kernel = np.array([[0.5, -1, 0.5], [-0.5, 0, 0.5], [0, 1, 0]]) # Note: This kernel might be simplified for 1D parabolic fit.
                                                                    # A standard 1D parabolic fit for 3 points (x-1, x, x+1)
                                                                    # uses coefficients derived directly from the values.

        # Perform sub-pixel refinement in the X direction using parabolic interpolation
        # Only perform if the peak is not at the border of the correlation map
        if 0 < x_peak < res.shape[1]-1:
            # Get the correlation values at (x_peak-1, x_peak, x_peak+1) along the y_peak row
            vec_x = np.array([res[y_peak, x_peak - 1], res[y_peak, x_peak], res[y_peak, x_peak + 1]]).reshape(3,1)
            
            # This 'kernel' application is a bit unusual for standard parabolic interpolation
            # For 1D parabolic interpolation given 3 points (x-1, y1), (x, y2), (x+1, y3):
            # The sub-pixel offset dx_sub = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
            # And the peak value is y2 - 0.25 * (y1 - y3)^2 / (y1 - 2*y2 + y3)
            # Let's use the standard formula for clarity and correctness.
            
            y_left, y_center, y_right = vec_x.flatten()
            denominator = y_left - 2 * y_center + y_right
            if denominator != 0: # Avoid division by zero
                dx_sub = (y_left - y_right) / (2 * denominator)
                # The refined peak value (coefficient)
                coeff_x = y_center - 0.25 * (y_left - y_right)**2 / denominator
            else: # If denominator is zero, it's a flat peak or linear, no parabolic refinement
                dx_sub = 0.0
                coeff_x = y_center # Use the integer peak value

        # Perform sub-pixel refinement in the Y direction using parabolic interpolation
        # Only perform if the peak is not at the border of the correlation map
        if 0 < y_peak < res.shape[0]-1:
            # Get the correlation values at (y_peak-1, y_peak, y_peak+1) along the x_peak column
            vec_y = np.array([res[y_peak - 1, x_peak],res[y_peak, x_peak],res[y_peak + 1, x_peak]]).reshape(3,1)
            
            y_up, y_center, y_down = vec_y.flatten()
            denominator = y_up - 2 * y_center + y_down
            if denominator != 0: # Avoid division by zero
                dy_sub = (y_up - y_down) / (2 * denominator)
                # The refined peak value (coefficient)
                coeff_y = y_center - 0.25 * (y_up - y_down)**2 / denominator
            else: # If denominator is zero, it's a flat peak or linear, no parabolic refinement
                dy_sub = 0.0
                coeff_y = y_center # Use the integer peak value

        # The overall matching coefficient is typically the product or geometric mean of the
        # refined coefficients in x and y. Using sqrt(coeff_x * coeff_y) is a common approach.
        coeffs[i] = float(np.sqrt(max(0, coeff_x * coeff_y))) # Ensure non-negative under sqrt

        # Calculate the final updated POI location in the current image.
        # This involves summing four components:
        # 1. x1, y1: The top-left corner of the search region (relative to curr_img's origin).
        # 2. x_peak, y_peak: The integer-pixel peak location within the 'search_region'.
        # 3. dx_sub, dy_sub: The sub-pixel offset from the integer peak.
        # 4. dx, dy: The original offset of the POI from its template's top-left corner.
        updated_templates[i] = [x1 + x_peak + dx_sub + dx, y1 + y_peak + dy_sub + dy]

    # Return the updated POI locations (np x 2 array) and their corresponding matching coefficients
    return updated_templates.reshape(-1,2), coeffs.reshape(-1)


def unit_test_01():
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    root = tk.Tk()
    root.withdraw()

    # img 1 is a 300 by 300 image, white backgrounded, 
    # with 2 black squares from (140, 140) to (150, 150) and from (150, 150) to (160, 160)
    img1 = np.ones((300, 300), dtype=np.uint8) * 255
    cv2.rectangle(img1, (140, 140), (150, 150), (0, 0, 0), -1)
    cv2.rectangle(img1, (150, 150), (160, 160), (0, 0, 0), -1)
    # img 2 is a 300 by 300 image, white backgrounded, 
    # with 2 black squares from (141, 142) to (151, 152) and from (151, 152) to (161, 162)
    img2 = np.ones((300, 300), dtype=np.uint8) * 255
    cv2.rectangle(img2, (141, 142), (151, 152), (0, 0, 0), -1)
    cv2.rectangle(img2, (151, 152), (161, 162), (0, 0, 0), -1)
    # Display the images
    cv2.imshow('Initial image (image 1)', img1); ikey= cv2.waitKey(0)
    cv2.imshow('Current image (image 2)', img2); ikey= cv2.waitKey(0)
    # destroy the windows
    cv2.destroyAllWindows()

    tmplt_input1 = "150.0 150.0 135 135 30 30"  # xi, yi, x0, y0, w, h
    tmplt_input2 = "160.0 160.0 150 150 20 20"  # xi, yi, x0, y0, w, h

    if tmplt_input1 is not None and tmplt_input2 is not None:
        # combine tmplt_input1 and tmplt_input2 to init_templates, which is a 2 by 6 array
        tmplt1_values = list(map(float, tmplt_input1.split()))
        tmplt2_values = list(map(float, tmplt_input2.split()))
        init_templates = np.array([tmplt1_values, tmplt2_values], dtype=np.float32).reshape(2, 6)

    (new_position, coeffs) = template_match_subpixels(img1, img2, init_templates)
    curr_pois = new_position

    print(f"Tracked point: {new_position[:,0:2]}")
    print(f"Matching coefficient: {coeffs}")

    # Optional visualization of the results by plotting two boxes on the current image
    img_plot = img2.copy()
    for i, (xi, yi) in enumerate(new_position):
        x0, y0, w, h = init_templates[i, 2:6].astype(int)
        x1 = int(xi - (init_templates[i,0]-x0))
        y1 = int(yi - (init_templates[i,1]-y0))
        cv2.rectangle(img_plot, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

    # 
    cv2.imshow('Tracked Points', img_plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def unit_test_02():
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    from imshow3 import imshow3 

    root = tk.Tk()
    root.withdraw()

    # use tk file dialog to select two images
    file1 = filedialog.askopenfilename(title="Select Initial Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    file2 = filedialog.askopenfilename(title="Select Current Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    img1 = cv2.imread(file1, cv2.IMREAD_COLOR_BGR)
    img2 = cv2.imread(file2, cv2.IMREAD_COLOR_BGR)

    # use imshow3 to ask user to select templates from img1
    from imshow3 import imshow3
    Xi, Xir, names = imshow3("Initial Image (image 1)", img1)
    
    # hstack Xi and Xir into init_templates
    init_templates = np.hstack((Xi, Xir)).astype(np.float32)

    # call template_match_subpixels
    curr_pois = init_templates[:, 0:2].copy()  # initial guess is the same as the initial templates
    (new_position, coeffs) = template_match_subpixels(img1, img2, init_templates, curr_pois)

    print(f"Initial points: \n{init_templates[:,0:6]}\n")
    print(f"Tracked points: \n{new_position[:,0:2]}\n")
    print(f"Matching coefficients: {coeffs}\n")

    # Save the initial templates (init_templates) to the worksheet "initial templates" of an xlsx file
    # with these columns: poi_name, xi, yi, x0, y0, w, h 
    import openpyxl
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "initial templates"
    ws.append(["poi_name", "xi", "yi", "x0", "y0", "w", "h"])
    for i in range(init_templates.shape[0]):
        ws.append([names[i], *init_templates[i, :]])
    # Save the tracked points (new_position) to the worksheet "tracked points" of the same xlsx file
    # with these columns: poi_name, xi, yi
    ws = wb.create_sheet(title="tracked points")
    ws.append(["poi_name", "xi", "yi", "coefficient"])
    for i in range(new_position.shape[0]):
        ws.append([names[i], *new_position[i, :2], coeffs[i]])
    # Save the workbook to a file
    # use tk file dialog to select the save location
    save_file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if save_file:
        wb.save(save_file)
        print(f"Results saved to {save_file}")

def run_gui_template_match_subpixels_image_sequence():
    # This function is designed to run the GUI for template matching sub-pixels
    # It uses tkinter to create a simple interface for users to specify:
    # 1. initial image
    # 2. image sequence to analyze 
    # 3. file of initial templates (poi_names, xi, yi, x0, y0, w, h)
    # 4. search_range_pixels, a scalar, 2-element list, or a NumPy array of shape (np_templates, 2)
    # 5. search_range_factors, a scalar, 2-element list, or a NumPy array of shape (np_templates, 2)
    from inputdlg3 import inputdlg3 
    import cv2 # don't know why but i need to import cv2 again here.
    # popup a dialog by inputdlg3 
    title = "Template Match for image sequence GUI"
    prompts = [
        "Initial Image (image file or a video file with a frame index):",
        "Image Sequence (image files or a video file):",
        "Definition Templates File (csv file path):",
        "Search Range Pixels (scalar, 2-element list, or np array of shape (np_templates, 2)):",
        "Search Range Factors (scalar, 2-element list, or np array of shape (np_templates, 2)):",
        "Tracked Points (csv file path):",
        "Output directory for drawn tracked results (optional):"
    ]
    datatypes = [
        "image",  # Initial Image
        "files",  # Image Sequence Directory
        "file",   # Initial Templates File
        "array -1 2",  # Search Range Pixels
        "array -1 2",  # Search Range Factors
        "filew",  # Tracked Points File
        "dir"     # Image Sequence Directory
    ]
    initvalues = [
        "c:/test/initial_image.jpg",  # Initial Image
        "",  # Image Sequence files
        "c:/test/templates.csv",  # Initial Templates File
        "",  # Search Range Pixels (default to 30 pixels)
        "2.0 0.5, 2.0 0.5, 2.0 0.5",  # Search Range Factors (default to 1.0)
        "c:/test/tracked_points.csv", # Tracked Points File
        "d:/test/drawn_results/"  # Output directory for drawn tracked results
    ]
    tooltips = [
        "Select the initial image file to use for template matching.\nIt can be an image, e.g., c:/a.jpg\nor a video file with frame index (1-base), e.g., c:/a.mp4 300",
        "Select image files or a video file.",
        "Select the csv file containing the initial templates (7 columns, each row has poi_names, xi, yi, x0, y0, w, h).",
        "Specify the search range in pixels as a scalar,\n  2-element list,\n  or a NumPy array of shape (np_templates, 2).",
        "If search range in pixels are empty, you need to specify the search range factors\n  as a scalar,\n  2-element list,\n  or a NumPy array of shape (np_templates, 2).",
        "Select a (new) csv file to save the tracked points.\nIt will be created if it does not exist.",
        "Select the output directory to save the drawn tracked results.\nIf not specified, no results will be drawn."
    ]
    # Call inputdlg3 to get user inputs
    while True:
        user_inputs = inputdlg3(title=title, prompts=prompts, datatypes=datatypes, initvalues=initvalues, tooltips=tooltips)
        # 
        if user_inputs is None:
            print("User cancelled the input dialog.")
            break
        # update initial value for possible next trial. When user input invalid data, this program pops up 
        # a message box and runs the input dialog again.
        initvalues = [
            user_inputs[0] if user_inputs is not None else initvalues[0],  # Initial Image
            user_inputs[1] if user_inputs is not None else initvalues[1],  # Image Sequence files
            user_inputs[2] if user_inputs is not None else initvalues[2],  # Initial Templates File
            user_inputs[3] if user_inputs is not None else initvalues[3],  # Search Range Pixels
            user_inputs[4] if user_inputs is not None else initvalues[4],  # Search Range Factors
            user_inputs[5] if user_inputs is not None else initvalues[5],  # Tracked Points File
            user_inputs[6] if user_inputs is not None else initvalues[6],  # Output directory for drawn results
        ]
        # Extract user inputs
        # init_img 
        init_img_path = user_inputs[0].strip().split()[0]  # Get the first part of the input
        if len(user_inputs[0].strip().split()) == 1:
            # if init_img_path has only one part, it is an image file
            init_img = cv2.imread(init_img_path, cv2.IMREAD_COLOR_BGR)
            # if init_img is None, popup error message and return 
            if init_img is None:
                return
        if len(user_inputs[0].strip().split()) >= 2:
            # if init_img_path has two parts, it is a video file with frame index
            frame_index = int(user_inputs[0].strip().split()[1])
            # read the frame from the video file
            import cv2
            cap = cv2.VideoCapture(init_img_path)
            # if fails to read the video frame, pops up an error message and try next iteration
            try:
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file: {init_img_path}")
                # get the initial frame (init_img) from the video file
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)  # Set the frame index (1-based)
                ret, init_img = cap.read()
                cap.release()  # Release the video capture object
            except:
                # popup a tk error message
                import tkinter as tk
                from tkinter import messagebox
                # without knowing whether there has been a tk window, popup a tk message dialog
                msg = f"Failed to open video file: {init_img_path}\nPlease check the file path and try again."
                if not tk._default_root:
                    root = tk.Tk(); root.withdraw()
                    messagebox.showerror("Error", msg)
                    root.destroy()
                else:
                    messagebox.showerror("Error", msg)
        # print a message that init_img is loaded
        print(f"Initial image loaded from: {init_img_path}. Resolution: {init_img.shape[1]}x{init_img.shape[0]} pixels.")
        # image sequence
        img_seq = user_inputs[1]
        if len(img_seq) == 0:
            # pop up a message box and return
            import tkinter as tk
            from tkinter import messagebox
            msg = "No image sequence files selected. Please select at least one image file or a video file."
            if not tk._default_root:
                # If there is no root window, create one
                root = tk.Tk(); root.withdraw()  # Hide the root window
                messagebox.showerror("Error", msg)
                root.destroy()
            else:
                messagebox.showerror("Error", msg)
            continue
        if len(img_seq) == 1:
            # if img_seq has only one part, only one file, very likely it is a video
            # check if it is a video file by trying to open it with cv2.VideoCapture
            import cv2
            vid = cv2.VideoCapture(img_seq[0])
            if not vid.isOpened():
                # if it is not a video file, pop up a message box and return
                import tkinter as tk
                from tkinter import messagebox
                msg = f"Failed to open video file: {img_seq[0]}\nPlease check the file path and try again."
                if not tk._default_root:
                # If there is no root window, create one
                    root = tk.Tk(); root.withdraw()
                    messagebox.showerror("Error", msg)
                    root.destroy()
                else:
                    messagebox.showerror("Error", msg)
                continue
            else:
                # if it is a video file, read the first frame
                num_images = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_images <= 0:
                    # if the video has no frames, pop up a message box and return
                    import tkinter as tk
                    from tkinter import messagebox
                    msg = f"The video file {img_seq[0]} has no frames. Please check the file path and try again."
                    if not tk._default_root:
                        root = tk.Tk(); root.withdraw()
                        messagebox.showerror("Error", msg)
                        root.destroy()
                    else:
                        messagebox.showerror("Error", msg)
                    continue
                vid.release()
        elif len(img_seq) > 1:
            # if img_seq has more than one part, it is a list of image files
            num_images = len(img_seq)
        # print message that the image sequence or a video is ready
        print(f"Image sequence or video file loaded: {img_seq[0]}. Total frames: {num_images}.")
        # load poi/template file
        init_templates_file = user_inputs[2].strip()
        # read the initial templates from the file
        from read_poi_7_columns import read_poi_7_columns
        poi_names, Xi, Xir = read_poi_7_columns(init_templates_file)
        init_templates = np.hstack((Xi, Xir)).astype(np.float32)
        # now we have: 
        #   init_img as the initial image (for templates)
        #   img_seq as the image sequence (list of image files or a video file, check the length of the list)
        #   poi_names, Xi, and Xir, for initial templates

        if len(img_seq) == 1:
            # if img_seq has only one part, it is a video file, so we can use cv2.VideoCapture to read the video
            import cv2
            vid = cv2.VideoCapture(img_seq[0])
            if not vid.isOpened():
                # if it is not a video file, pop up a message box and return
                import tkinter as tk
                from tkinter import messagebox
                msg = f"Failed to open video file: {img_seq[0]}\nPlease check the file path and try again."
                if not tk._default_root:
                    root = tk.Tk(); root.withdraw()
                    messagebox.showerror("Error", msg)
                    root.destroy()
                else:
                    messagebox.showerror("Error", msg)
                continue
            # end if vid.isOpened()
        # end if len(img_seq) == 1

        # create an empty array to store the tracked points 
        num_pois = init_templates.shape[0]  # Number of templates to track
        xi_result = np.ones((num_images, init_templates.shape[0] * 3), dtype=np.float32) * np.nan # shape: (num_images, np_templates, 2)
        # make xi_header ['frame_index', 'xi_name1', 'yi_name1', 'tmcoeff_name1', 'xi_name2', 'yi_name2', 'tmcoeff_name2', ...]
        # where name1, name2, ... are the names of the POIs
        xi_header = ["frame_index"]  # Header for the CSV file
        for name in poi_names:
            xi_header.extend([f"xi_{name}", f"yi_{name}", f"tmcoeff_{name}"])
#        xi_header = ["frame_index,"] + [f"xi_{name}, yi_{name}, tmcoeff_{name}".split(',') for name in poi_names]  # Header for the CSV file
        # Start timing the loop
        start_time_loop = cv2.getTickCount() 
        # read an image from vid and track
        for iframe in range(num_images):
            # for each frame in the image sequence, store the image at curr_img
            if (len(img_seq) == 1):
                # if the image is from a video file, read the next frame
                ret, curr_img = vid.read()
                if ret == False:
                    curr_img = None
            else:
                # if the image is from a list of image files, read the next image
                curr_img = cv2.imread(img_seq[iframe], cv2.IMREAD_COLOR_BGR)
            # end if len(img_seq) == 1
            # template match with subpixels
            # init_template is hstack of Xi and Xir 
            init_templates = np.hstack((Xi, Xir)).astype(np.float32)
            if iframe == 0:
                guess_poi = init_templates[:, 0:2].copy()  # Initial guess is the same as the initial templates
            elif iframe >= 1:
                for ipoi in range(num_pois):
                    # update the initial templates with the previous frame's tracked points
                    guess_poi[ipoi, 0:2] = xi_result[iframe - 1, ipoi * 3 + 0:ipoi * 3 + 2]
            else: # iframe >= 2
                # linear motion guess
                for ipoi in range(num_pois):
                    # update the initial templates with the previous frame's tracked points
                    guess_poi[ipoi, 0:2] = (-1) * xi_result[iframe - 2, ipoi * 3 + 0:ipoi * 3 + 2] \
                                         +   2  * xi_result[iframe - 1, ipoi * 3 + 0:ipoi * 3 + 2]
            # end if iframe == 0
            tracked_pois, coeffs = template_match_subpixels(
                init_img=init_img, 
                curr_img=curr_img, 
                init_templates=np.array(init_templates, dtype=float),  # Ensure init_templates is a NumPy array of float32
                curr_pois=np.array(guess_poi, dtype=float), # Use the initial POI locations as the current guess
                search_range_pixels=None,  # Use default search range pixels
                search_range_factors=1.0
            )
            # save result to xi_result
            for ipoi in range(num_pois):
                xi_result[iframe, ipoi * 3 + 0] = tracked_pois[ipoi, 0]  # xi
                xi_result[iframe, ipoi * 3 + 1] = tracked_pois[ipoi, 1]  # yi
                xi_result[iframe, ipoi * 3 + 2] = coeffs[ipoi]  # tmcoeff
            # draw templates on the current image
            # if user_inputs[5] is a valid directory, draw the tracked points on the current image
            # the file name is the original image file name with '_tracked' suffix
            # for example, if the basename of the image file is 'image1.jpg', the tracked image will be 'image1_tracked.jpg'
            try: # try to draw the tracked points on a new image 
                import os
                from drawTemplates3 import drawTemplates3
                # if user_inputs[6] is a valid directory, draw the tracked points on the current image
                if len(user_inputs) > 6 and user_inputs[6].strip() and os.path.isdir(user_inputs[6].strip()):
                    output_dir = user_inputs[6].strip()
                    # get the basename of the current image file
                    if len(img_seq) == 1:
                        # if it is a video file, use the video file name
                        # for example, if the video file is 'video.mp4', the output video file will be 'video_tracked_frame_%06d.mp4'
                        output_image_filename_base = os.path.basename(img_seq[0]) + "_tracked_frame_%06d.JPG" % iframe 
                    else:
                        # if it is a list of image files, use the current image file name
                        # for example, if the image file is 'image1.jpg', the output image file will be 'image1_tracked.jpg'
                        output_image_filename_base = os.path.basename(img_seq[iframe]) + "_tracked" + os.path.splitext(img_seq[iframe])[1]
                    # end if len(img_seq) == 1
                    # draw the tracked points on the current image
                    #   def drawTemplates3(image: np.ndarray, poi=None, poir=None, 
                    #       poi_names: list = None, draw_options: dict = None) -> np.ndarray:
                    # updated_poir is the updated poi region (rectangles), 
                    # which size is (num_pois, 4), integers, that describes the bounding boxes of the new POIs, for example:
                    # if  init_templates[0] is [150.0, 150.0, 135, 135, 30, 30], 
                    # and curr_pois[0] is [151.2, 152.6],
                    # then the updated_poir[0] is [135 + round(151.2 - 150.0), 135 + round(152.6 - 150.0), 30, 30])
                    updated_poi_poir = np.zeros((num_pois, 4), dtype=int)
                    poi_names_show = poi_names.copy()
                    for ipoi in range(num_pois):
                        # get the current POI location
                        xi, yi = tracked_pois[ipoi]
                        # get the initial template bounding box
                        x0, y0, w, h = init_templates[ipoi, 2:6].astype(int)
                        # calculate the new bounding box based on the current POI location
                        updated_poi_poir[ipoi] = [x0 + round(xi - init_templates[ipoi, 0]), 
                                                   y0 + round(yi - init_templates[ipoi, 1]), w, h]
                        # update the poi_names with original poi_names + coefficients
                        poi_names_show[ipoi] = f"{poi_names[ipoi]} ({coeffs[ipoi]:.3f})"
                    drawn_img = drawTemplates3(
                        image=curr_img, 
                        poi=tracked_pois,
                        poir=updated_poi_poir, 
                        poi_names=poi_names_show,  
                    )
                    # end for ipoi in range(num_pois)
                    # save the current image with tracked points to the output directory
                    output_image_path = os.path.join(output_dir, output_image_filename_base)
                    cv2.imwrite(output_image_path, drawn_img)
                # end if len(user_inputs) > 5 and user_inputs[5].strip() and os.path.isdir(user_inputs[5].strip())
            except:
                if iframe == 0:
                    print("Error: Failed to draw tracked points on the current image. "
                          "Please check if the output directory is valid and writable.")
                pass
            # end of try to draw the tracked points on a new image
            # check current time 
            curr_time = cv2.getTickCount()
            # print message that the current frame is processed
            # print frame index / total number of frames, progress percentage, elapsed time, ETA time
            elapsed_time = (curr_time - start_time_loop) / cv2.getTickFrequency()
            eta_time = (elapsed_time / (iframe + 1)) * (num_images - (iframe + 1))
            progress_percentage = (iframe + 1) / num_images * 100
            print("\b"*500, end="")  # Clear the line for better readability
            print(f"Frame {iframe + 1}/{num_images} ({progress_percentage:.2f}%) - "
                  f"Elapsed time: {elapsed_time:.2f}s, ETA: {eta_time:.2f}s", end="")

#            print(f"Frame {iframe + 1}/{num_images}: ", end="")
#            for ipoi in range(num_pois):
#                print(f"{poi_names[ipoi]}: ({updated_templates[ipoi, 0]:.2f}, {updated_templates[ipoi, 1]:.2f}), "
#                      f"tmcoeff: {coeffs[ipoi]:.4f}", end=", ")

        # end of for iframe in range(num_images)
        # save the tracked points to a csv file
        tracked_points_file = user_inputs[5].strip()
        # use csv module to save the tracked points (we do not use pandas because we prefer small and easy-to-maintain module)
        import csv
        with open(tracked_points_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(xi_header)
            # write the data
            for iframe in range(num_images):
                row = [iframe + 1]  # Frame index (1-based)
                for ipoi in range(num_pois):
                    row.append(xi_result[iframe, ipoi * 3 + 0]) 
                    row.append(xi_result[iframe, ipoi * 3 + 1])
                    row.append(xi_result[iframe, ipoi * 3 + 2])
                writer.writerow(row)
        # end of saving tracked points to a csv file
        # print message that the tracked points are saved
        print(f"\nTracked points saved to: {tracked_points_file}. Number of frames: {num_images}. Number of templates: {num_pois}.")
        break


if __name__ == "__main__":
    # To run unit_test_01, uncomment the line below and comment out unit_test_02()
    # unit_test_01()
    # To run unit_test_02, uncomment the line below and ensure you have 'imshow3' available
    #unit_test_02()
    # gui 
    run_gui_template_match_subpixels_image_sequence()