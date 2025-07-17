"""
imshow4(inputs: dict) -> dict

This function allows users to interactively define multiple Points of Interest (POIs) on an image and select template regions (ROIs) for each POI. It's an enhanced version of imshow3, adapted for modular tool environments like ImTools. 

Inputs (all packed in a single dictionary):
- 'img': numpy.ndarray – the image to display.
- 'Xi': optional, pre-defined POIs as a numpy array (Nx2).
- 'Xir': optional, pre-defined template regions as numpy array (NxHxWx3).
- 'poi_names': optional list of POI names.
- 'winname': string – window title shown in cv2.imshow.
- 'winmax': (width, height) tuple – maximum size for display window.
- 'interp': cv2 interpolation constant – for resizing display image.

Returns (dictionary):
{
    'Xi': POI coordinates (Nx2 numpy array),
    'Xir': POI ROIs (NxHxWx3 numpy array),
    'poi_names': list of POI names
}

----

Examples:

1. Minimal usage:
imshow4({'img': image})

2. Custom window title and size:
imshow4({'img': image, 'name_prefix': 'sceneA', 'winname': 'Select POIs', 'winmax': (1024,768)})

3. With interpolation method:
imshow4({'img': image, 'name_prefix': 'zoomed', 'interp': cv2.INTER_AREA})

4. Load previous POIs:
imshow4({'img': image, 'Xi': old_Xi, 'Xir': old_Xir, 'poi_names': old_names, 'name_prefix': 'refined'})

5. From tkinter file dialog:
img_path = filedialog.askopenfilename()
img = cv2.imread(img_path)
imshow4({'img': img, 'name_prefix': 'dialog'})

----
"""
import cv2
from imshow3 import imshow3

def imshow4(inputs: dict) -> dict:
    img = inputs['img']
    Xi = inputs.get('Xi', None)
    Xir = inputs.get('Xir', None)
    poi_names = inputs.get('poi_names', None)
    winname = inputs.get('winname', 'imshow4')
    winmax = inputs.get('winmax', (1280, 720))
    interp = inputs.get('interp', cv2.INTER_NEAREST)

    Xi_out, Xir_out, poi_names_out = imshow3(
        img=img,
        Xi=Xi,
        Xir=Xir,
        poi_names=poi_names,
        winname=winname,
        winmax=winmax,
        interp=interp
    )

    return {
        f'{name_prefix}_Xi': Xi_out,
        f'{name_prefix}_Xir': Xir_out,
        f'{name_prefix}_names': poi_names_out
    }

