from opendm import log
from opendm.thermal_tools import dji_unpack
import cv2
import os

def resize_to_match(image, match_photo = None):
    """
    Resize images to match the dimension of another photo
    :param image  numpy array containing image data to resize
    :param match_photo ODM_Photo whose dimensions should be used for resize
    :return numpy array with resized image data
    """
    if match_photo is not None:
        h, w, _ = image.shape
        if w != match_photo.width or h != match_photo.height:
            image = cv2.resize(image, None, 
                    fx=match_photo.width/w, 
                    fy=match_photo.height/h,
                    interpolation=cv2.INTER_LANCZOS4)
    return image

def dn_to_temperature(photo, image, dataset_tree=None):
    """
    Convert Digital Number values to temperature (C) values
    :param photo ODM_Photo
    :param image numpy array containing image data
    :param dataset_tree path to original source image to read data using PIL for DJI thermal photos
    :return numpy array with temperature (C) image values
    """   

    # Handle thermal bands
    if photo.is_thermal():
        # Every camera stores thermal information differently
        if photo.camera_make == "MicaSense": # Assume MicaSense cameras (RedEdge-MX, Altum, Altum-PT, and others) are using the same thermal calibration
            image = image.astype("float32")
            image -= (273.15 * 100.0) # Convert Kelvin to Celsius
            image *= 0.01
            return image
        elif photo.camera_make == "DJI" and photo.camera_model == "ZH20T":            
            filename, file_extension = os.path.splitext(photo.filename)
            # DJI H20T high gain mode supports measurement of -40~150 celsius degrees
            if file_extension.lower() in [".tif", ".tiff"] and image.min() >= 23315: # Calibrated grayscale tif
                image = image.astype("float32")
                image -= (273.15 * 100.0) # Convert Kelvin to Celsius
                image *= 0.01
                return image
            else:
                return image
        elif photo.camera_make == "DJI" and photo.camera_model == "MAVIC2-ENTERPRISE-ADVANCED":
            try:
                image = dji_unpack.extract_temperatures_dji(photo, image, dataset_tree)
                image = image.astype("float32")
            except Exception as e:
                pass
            return image
        else:
            log.ODM_WARNING("Unsupported camera [%s %s], thermal band will have digital numbers." % (photo.camera_make, photo.camera_model))
            return image
    else:
        # image = image.astype("float32")
        log.ODM_WARNING("Tried to radiometrically calibrate a non-thermal image with temperature values (%s)" % photo.filename)
        return image

