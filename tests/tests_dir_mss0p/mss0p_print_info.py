
# Given four dictionaries containing information about a mss0p project:
# basic_info, camera_parameters, pois_definition, and image_sources,

# The basic_info dictionary looks like this:
#   basic_info = {'num_cameras': 4, 'num_pois': 10, 'num_steps': 100}
#
# The camera_paramters dictionary looks like this:
# camera_parameters = {
#     'camera_name_1': {
#         'image_size': (1920, 1080),
#         'rvec': np.array([0.0, 0.0, 0.0]).reshape(3,1),  # rotation vector as a 3x1 array
#         'tvec': np.array([0.0, 0.0, 0.0]).reshape(3,1),  # translation vector as a 3x1 array
#         'cmat': np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]).reshape(3, 3),
#         'dvec': np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)  # or more elements if available
#     }
#    'camera_name_2': {
#         'image_size': (1920, 1080),
#         'rvec': np.array([0.0, 0.0, 0.0]).reshape(3,1),  # rotation vector as a 3-element array
#         'tvec': np.array([0.0, 0.0, 0.0]).reshape(3,1),  # translation vector as a 3-element array
#         'cmat': np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]).reshape(3, 3),
#         'dvec': np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)  # or more elements if available
# #   }
#
# # The data format of pois_definition dictionary is:
#  pois_definition = {
#      'poi_name_1': {
#          'Xw': (1.0, 2.0, 3.0),  # world coordinates of the poi
#          'Xi': {
#              'camera_name_1': (100.0, 200.0),  # image coordinates of the poi in camera 1
#              'camera_name_2': (150.0, 250.0),  # image coordinates of the poi in camera 2
#          },
#          'Xir': {
#              'camera_name_1': (50, 50, 100, 100),  # template in camera 1
#              'camera_name_2': (60, 60, 120, 120),  # template in camera 2
#          }
#      }
#      'poi_name_2': {
#          'Xw': (4.0, 5.0, 6.0),  # world coordinates of the poi
#          'Xi': {
#              'camera_name_1': (110.0, 210.0),  # image coordinates of the poi in camera 1
#              'camera_name_2': (160.0, 260.0),  # image coordinates of the poi in camera 2
#          },
#          'Xir': {
#              'camera_name_1': (60, 60, 120, 120),  # template in camera 1
#              'camera_name_2': (70, 70, 140, 140),  # template in camera 2
#          }
#      }
#  }
#
#  The image_sources dictionary looks like this:
#  image_sources = {
#      'camera_name_1': '/path/to/video1.mp4',
#      'camera_name_2': '/path/to/video2.mp4',
#      'camera_name_3': ['/path/to/camera_name_3/image1.jpg', '/path/to/camera_name_3/image2.jpg', ...]
#      'camera_name_4': ['/path/to/camera_name_4/image1.jpg', '/path/to/camera_name_4/image2.jpg', ...]
#  }



def basic_info_to_text(basic_info=None):
    if basic_info is None:
        return "# No basic information provided.\n" 
    output = ""
    output += "=== Basic Information ===\n"
    for key, value in basic_info.items():
        if isinstance(value, list):
            value_str = ', '.join(map(str, value))
        elif isinstance(value, dict):
            value_str = ', '.join(f"{k}: {v}" for k, v in value.items())
        else:
            value_str = str(value)
        output += f"  basic_info[\"{key}\"] = {value_str}\n"    
    return output

def camera_parameters_to_text(camera_parameters=None):
    if camera_parameters is None:
        return "# No camera parameters provided.\n"
    output = ""
    output += "=== Camera Parameters ===\n"
    # for each camera
    output += f"  Number of cameras: {len(camera_parameters)}\n"
    for cam_name in camera_parameters.keys():
        output += f"  Camera name: {cam_name}\n"
        output += f"   camera_parameters[\"{cam_name}\"][\"image_size\"] = {camera_parameters[cam_name]['image_size']}\n"
        output += f"   camera_parameters[\"{cam_name}\"][\"rvec\"] is {str(camera_parameters[cam_name]['rvec'].T)}.T\n"
        output += f"   camera_parameters[\"{cam_name}\"][\"tvec\"] is {str(camera_parameters[cam_name]['tvec'].T)}.T\n"
        output += f"   camera_parameters[\"{cam_name}\"][\"cmat\"] is\n {str(camera_parameters[cam_name]['cmat'])}\n"
        output += f"   camera_parameters[\"{cam_name}\"][\"dvec\"] is {str(camera_parameters[cam_name]['dvec'].T)}.T\n"
    return output

def pois_definition_to_text(pois_definition=None):
    if pois_definition is None:
        return "# No points of interest (POIs) definition provided.\n"
    output = ""
    output += "=== Points of Interest (POIs) Definition ===\n"
    output += f"  Number of POIs: {len(pois_definition)}\n"
    for poi_name, poi_data in pois_definition.items():
        output += f"  POI name: {poi_name}\n"
        output += f"   pois_definition[\"{poi_name}\"][\"Xw\"] = {str(poi_data['Xw'])}\n"
        for cam_name, coords in poi_data['Xi'].items():
            output += f"   pois_definition[\"{poi_name}\"][\"Xi\"][\"{cam_name}\"]: {str(coords)}\n"
        for cam_name, roi in poi_data['Xir'].items():
            output += f"   pois_definition[\"{poi_name}\"][\"Xir\"][\"{cam_name}\"]: {str(roi)}\n"
    return output

def image_sources_to_text(image_sources=None):
    if image_sources is None:
        return "# No image sources provided.\n"
    output = ""
    output += "=== Image Sources ===\n"
    output += f"  Number of cameras: {len(image_sources)}\n"
    for cam_name, sources in image_sources.items():
        # if the source is a single string, that is a video file, then print this:
        #    image_sources["camera_name_1"] = "/path/to/video1.mp4"
        # if the source is a list, that is a list of image files, then print the 
        #    first file ane the last file in the list, like this:
        #    image_sources["camera_name_2"] = ["/path/to/camera_name_2/image1.jpg", ..., "/path/to/camera_name_2/imageN.jpg"]
        if isinstance(sources, str):
            output += f"  image_sources[\"{cam_name}\"] = \"{sources}\"\n"
        elif isinstance(sources, list):
            if len(sources) >=2:
                output += f"  image_sources[\"{cam_name}\"] = [{sources[0]}, ..., {sources[-1]}]\n"
            elif len(sources) == 1:
                output += f"  image_sources[\"{cam_name}\"] = [{sources[0]}]\n"
            else:
                output += f"  image_sources[\"{cam_name}\"] = []\n"
        else:
            output += f"  image_sources[\"{cam_name}\"] = {str(sources)}\n"
    return output


def info_to_text(basic_info=None, camera_parameters=None, pois_definition=None, 
               image_sources=None, project_file_path=None, print_widget=None):
    output = ""
    output += basic_info_to_text(basic_info)
    output += camera_parameters_to_text(camera_parameters)
    output += pois_definition_to_text(pois_definition)
    output += image_sources_to_text(image_sources)
    return output


