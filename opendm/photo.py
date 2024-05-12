import logging
import re
import os
import base64
import struct
import cv2
import math

import exifread
import numpy as np
from six import string_types
from datetime import datetime, timedelta, timezone
import pytz

from opendm import io
from opendm import log
from opendm import system
from opendm.rollingshutter import get_rolling_shutter_readout
import xmltodict as x2d
from opendm import get_image_size
from xml.parsers.expat import ExpatError
from opensfm.sensors import sensor_data
from opensfm.geo import ecef_from_lla

projections = ['perspective', 'fisheye', 'brown', 'dual', 'equirectangular', 'spherical']

def find_largest_photo_dims(photos):
    max_mp = 0
    max_dims = None

    for p in photos:
        if p.width is None or p.height is None:
            continue
        mp = p.width * p.height
        if mp > max_mp:
            max_mp = mp
            max_dims = (p.width, p.height)

    return max_dims

def find_largest_photo_dim(photos):
    max_dim = 0
    for p in photos:
        if p.width is None:
            continue
        max_dim = max(max_dim, max(p.width, p.height))

    return max_dim

def find_largest_photo(photos):
    max_p = None
    max_area = 0
    for p in photos:
        if p.width is None:
            continue
        area = p.width * p.height

        if area > max_area:
            max_area = area
            max_p = p

    return max_p

def get_mm_per_unit(resolution_unit):
    """Length of a resolution unit in millimeters.

    Uses the values from the EXIF specs in
    https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html

    Args:
        resolution_unit: the resolution unit value given in the EXIF
    """
    if resolution_unit == 2:  # inch
        return 25.4
    elif resolution_unit == 3:  # cm
        return 10
    elif resolution_unit == 4:  # mm
        return 1
    elif resolution_unit == 5:  # um
        return 0.001
    else:
        log.ODM_WARNING("Unknown EXIF resolution unit value: {}".format(resolution_unit))
        return None

class PhotoCorruptedException(Exception):
    pass

class GPSRefMock:
    def __init__(self, ref):
        self.values = [ref]


class ODM_Photo:
    """ODMPhoto - a class for ODMPhotos"""

    def __init__(self, path_file):
        self.filename = os.path.basename(path_file)
        self.mask = None

        # Standard tags (virtually all photos have these)
        self.width = None
        self.height = None
        self.camera_make = ''
        self.camera_model = ''
        self.orientation = 1

        # Geo tags
        self.latitude = None
        self.longitude = None
        self.altitude = None

        # Multi-band fields
        self.band_name = 'RGB'
        self.band_index = 0
        self.rig_relatives = None
        self.capture_uuid = None

        # Multi-spectral fields
        self.fnumber = None
        self.radiometric_calibration = None
        self.black_level = None
        self.gain = None
        self.gain_adjustment = None

        # Capture info
        self.exposure_time = None
        self.iso_speed = None
        self.bits_per_sample = None
        self.vignetting_center = None
        self.vignetting_polynomial = None
        self.distortion_parameters = None
        self.principal_point = None
        self.focal_plane_resolution_x = None
        self.focal_plane_resolution_y = None
        self.utc_time = None

        # DLS
        self.sun_sensor = None
        self.dls_yaw = None
        self.dls_pitch = None
        self.dls_roll = None

        # Irradiance calibration
        self.sensor_model_coefficient = None
        self.spectral_irradiance = None
        self.horizontal_irradiance = None
        self.irradiance_scale_to_si = None
        self.irradiance_list = None
        self.irradiance_calibration = None

        # OPK angles
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.omega = None
        self.phi = None
        self.kappa = None

        # Aircraft speed
        self.speed_x = None
        self.speed_y = None
        self.speed_z = None

        # Original image width/height at capture time (before possible resizes)
        self.exif_width = None
        self.exif_height = None

        # self.center_wavelength = None
        # self.bandwidth = None

        # RTK
        self.gps_xy_stddev = None # Dilution of Precision X/Y
        self.gps_z_stddev = None # Dilution of Precision Z

        # Misc SFM
        self.camera_projection = 'brown'
        self.focal_ratio = 0.85
        self.focal_length = None

        # parse values from metadata
        self.parse_exif_values(path_file)

    def __str__(self):
        return '{} | camera: {} {} | dimensions: {} x {} | lat: {} | lon: {} | alt: {} | band: {} ({})'.format(
                            self.filename, self.camera_make, self.camera_model, self.width, self.height,
                            self.latitude, self.longitude, self.altitude, self.band_name, self.band_index)

    def set_mask(self, mask):
        self.mask = mask

    def update_with_geo_entry(self, geo_entry):
        self.latitude = geo_entry.y
        self.longitude = geo_entry.x
        self.altitude = geo_entry.z
        if geo_entry.yaw is not None and geo_entry.pitch is not None and geo_entry.roll is not None:
            self.yaw = geo_entry.yaw
            self.pitch = geo_entry.pitch
            self.roll = geo_entry.roll
            self.dls_yaw = geo_entry.yaw
            self.dls_pitch = geo_entry.pitch
            self.dls_roll = geo_entry.roll
        self.gps_xy_stddev = geo_entry.horizontal_accuracy
        self.gps_z_stddev = geo_entry.vertical_accuracy

    def parse_exif_values(self, _path_file):
        # Disable exifread log
        logging.getLogger('exifread').setLevel(logging.CRITICAL)

        try:
            self.width, self.height = get_image_size.get_image_size(_path_file)
        except Exception as e:
            raise PhotoCorruptedException(str(e))

        tags = {}
        xtags = {}

        with open(_path_file, 'rb') as f:
            tags = exifread.process_file(f, details=True, extract_thumbnail=False)
            try:
                if 'Image Make' in tags:
                    try:
                        self.camera_make = tags['Image Make'].values
                        self.camera_make = self.camera_make.strip()
                    except UnicodeDecodeError:
                        log.ODM_WARNING("EXIF Image Make might be corrupted")
                        self.camera_make = "unknown"
                if 'Image Model' in tags:
                    try:
                        self.camera_model = tags['Image Model'].values
                        self.camera_model = self.camera_model.strip()
                    except UnicodeDecodeError:
                        log.ODM_WARNING("EXIF Image Model might be corrupted")
                        self.camera_model = "unknown"
                if 'GPS GPSAltitude' in tags:
                    self.altitude = self.float_value(tags['GPS GPSAltitude'])
                    if 'GPS GPSAltitudeRef' in tags and self.int_value(tags['GPS GPSAltitudeRef']) is not None and self.int_value(tags['GPS GPSAltitudeRef']) > 0:
                        self.altitude *= -1
                if 'GPS GPSLatitude' in tags and 'GPS GPSLatitudeRef' in tags:
                    self.latitude = self.dms_to_decimal(tags['GPS GPSLatitude'], tags['GPS GPSLatitudeRef'])
                elif 'GPS GPSLatitude' in tags:
                    log.ODM_WARNING("GPS position for %s might be incorrect, GPSLatitudeRef tag is missing (assuming N)" % self.filename)
                    self.latitude = self.dms_to_decimal(tags['GPS GPSLatitude'], GPSRefMock('N'))
                if 'GPS GPSLongitude' in tags and 'GPS GPSLongitudeRef' in tags:
                    self.longitude = self.dms_to_decimal(tags['GPS GPSLongitude'], tags['GPS GPSLongitudeRef'])
                elif 'GPS GPSLongitude' in tags:
                    log.ODM_WARNING("GPS position for %s might be incorrect, GPSLongitudeRef tag is missing (assuming E)" % self.filename)
                    self.longitude = self.dms_to_decimal(tags['GPS GPSLongitude'], GPSRefMock('E'))
                if 'Image Orientation' in tags:
                    self.orientation = self.int_value(tags['Image Orientation'])
            except (IndexError, ValueError) as e:
                log.ODM_WARNING("Cannot read basic EXIF tags for %s: %s" % (self.filename, str(e)))

            try:
                if 'Image Tag 0xC61A' in tags:
                    self.black_level = self.list_values(tags['Image Tag 0xC61A'])
                elif 'BlackLevel' in tags:
                    self.black_level = self.list_values(tags['BlackLevel'])
                elif 'Image BlackLevel' in tags:
                    self.black_level = self.list_values(tags['Image BlackLevel'])

                if 'EXIF ExposureTime' in tags:
                    self.exposure_time = self.float_value(tags['EXIF ExposureTime'])

                if 'EXIF FNumber' in tags:
                    self.fnumber = self.float_value(tags['EXIF FNumber'])

                if 'EXIF ISOSpeed' in tags:
                    self.iso_speed = self.int_value(tags['EXIF ISOSpeed'])
                elif 'EXIF PhotographicSensitivity' in tags:
                    self.iso_speed = self.int_value(tags['EXIF PhotographicSensitivity'])
                elif 'EXIF ISOSpeedRatings' in tags:
                    self.iso_speed = self.int_value(tags['EXIF ISOSpeedRatings'])

                if 'Image BitsPerSample' in tags:
                    self.bits_per_sample = self.int_value(tags['Image BitsPerSample'])

                if 'EXIF DateTimeOriginal' in tags:
                    str_time = tags['EXIF DateTimeOriginal'].values
                    utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
                    subsec = 0
                    if 'EXIF SubSecTime' in tags:
                        subsec = self.int_value(tags['EXIF SubSecTime'])
                    negative = 1.0
                    if subsec < 0:
                        negative = -1.0
                        subsec *= -1.0
                    subsec = float('0.{}'.format(int(subsec)))
                    subsec *= negative
                    ms = subsec * 1e3
                    utc_time += timedelta(milliseconds = ms)
                    timezone = pytz.timezone('UTC')
                    epoch = timezone.localize(datetime.utcfromtimestamp(0))
                    self.utc_time = (timezone.localize(utc_time) - epoch).total_seconds() * 1000.0

                if 'MakerNote SpeedX' in tags and \
                    'MakerNote SpeedY' in tags and \
                    'MakerNote SpeedZ' in tags:
                    self.speed_x = self.float_value(tags['MakerNote SpeedX'])
                    self.speed_y = self.float_value(tags['MakerNote SpeedY'])
                    self.speed_z = self.float_value(tags['MakerNote SpeedZ'])

                if 'EXIF ExifImageWidth' in tags and \
                   'EXIF ExifImageLength' in tags:
                   self.exif_width = self.int_value(tags['EXIF ExifImageWidth'])
                   self.exif_height = self.int_value(tags['EXIF ExifImageLength'])

                if 'EXIF FocalPlaneXResolution' in tags:
                   self.focal_plane_resolution_x = self.float_value(tags['EXIF FocalPlaneXResolution'])

                if 'EXIF FocalPlaneYResolution' in tags:
                   self.focal_plane_resolution_y = self.float_value(tags['EXIF FocalPlaneYResolution'])

                if 'EXIF FocalLength' in tags:
                   self.focal_length = self.float_value(tags['EXIF FocalLength'])

            except Exception as e:
                log.ODM_WARNING("Cannot read extended EXIF tags for %s: %s" % (self.filename, str(e)))

            # Warn if GPS coordinates are suspiciously wrong
            if self.latitude is not None and self.latitude == 0 and \
                self.longitude is not None and self.longitude == 0:
                log.ODM_WARNING("%s has GPS position (0,0), possibly corrupted" % self.filename)


            # Extract XMP tags
            f.seek(0)
            xmp = self.get_xmp(f)

            for xtags in xmp:
                try:
                    band_name = self.get_xmp_tag(xtags, ['Camera:BandName', '@Camera:BandName'])
                    if band_name is not None:
                        self.band_name = band_name.replace(" ", "")

                    self.set_attr_from_xmp_tag('sensor_model_coefficient', xtags, [
                        'Camera:SensorModel',
                    ])

                    self.set_attr_from_xmp_tag('band_index', xtags, [
                        'DLS:SensorId', # Micasense RedEdge
                        '@Camera:RigCameraIndex', # Parrot Sequoia, Sentera 21244-00_3.2MP-GS-0001
                        'Camera:RigCameraIndex', # MicaSense Altum
                    ])

                    self.set_attr_from_xmp_tag('rig_relatives', xtags, [
                        'Camera:RigRelatives' # MicaSense Altum and RedEdge-MX
                    ])

                    self.set_attr_from_xmp_tag('radiometric_calibration', xtags, [
                        'MicaSense:RadiometricCalibration',
                    ])

                    self.set_attr_from_xmp_tag('vignetting_center', xtags, [
                        'Camera:VignettingCenter',
                        'Sentera:VignettingCenter',
                    ])

                    self.set_attr_from_xmp_tag('vignetting_polynomial', xtags, [
                        'Camera:VignettingPolynomial',
                        'Sentera:VignettingPolynomial',
                    ])

                    self.set_attr_from_xmp_tag('principal_point', xtags, [
                        'Camera:PrincipalPoint' # Micasense Altum and RedEdge-MX
                    ])

                    self.set_attr_from_xmp_tag('distortion_parameters', xtags, [
                        'Camera:PerspectiveDistortion' # Micasense Altum and RedEdge-MX
                    ])

                    self.set_attr_from_xmp_tag('focal_length', xtags, [
                        'Camera:PerspectiveFocalLength' # Micasense Altum and RedEdge-MX
                    ], float)

                    self.set_attr_from_xmp_tag('horizontal_irradiance', xtags, [
                        'Camera:HorizontalIrradiance',
                        'DLS:HorizontalIrradiance' # Micasense DLS2
                    ], float)

                    self.set_attr_from_xmp_tag('irradiance_scale_to_si', xtags, [
                        'Camera:IrradianceScaleToSIUnits'
                    ], float)

                    self.set_attr_from_xmp_tag('sun_sensor', xtags, [
                        'Camera:SunSensor',
                    ], float)

                    self.set_attr_from_xmp_tag('irradiance_calibration', xtags, [
                        '@Camera:IrradianceCalibrationMeasurement',   # Parrot Sequoia
                    ])

                    self.set_attr_from_xmp_tag('irradiance_list', xtags, [
                        'Camera:IrradianceList',   # Parrot Sequoia
                    ])

                    self.set_attr_from_xmp_tag('spectral_irradiance', xtags, [
                        'Camera:SpectralIrradiance',
                        'Camera:Irradiance',
                        'DLS:SpectralIrradiance' # Micasense DLS2
                    ], float)

                    self.set_attr_from_xmp_tag('capture_uuid', xtags, [
                        '@drone-dji:CaptureUUID', # DJI
                        'MicaSense:CaptureId', # MicaSense Altum
                        '@Camera:ImageUniqueID', # sentera 6x
                        '@Camera:CaptureUUID', # Parrot Sequoia
                    ])

                    self.set_attr_from_xmp_tag('gain', xtags, [
                        '@drone-dji:SensorGain'
                    ], float)

                    self.set_attr_from_xmp_tag('gain_adjustment', xtags, [
                        '@drone-dji:SensorGainAdjustment'
                    ], float)

                    # Camera make / model for some cameras is stored in the XMP
                    if self.camera_make == '':
                        self.set_attr_from_xmp_tag('camera_make', xtags, [
                            '@tiff:Make'
                        ])
                    if self.camera_model == '':
                        self.set_attr_from_xmp_tag('camera_model', xtags, [
                            '@tiff:Model'
                        ])

                    # DJI GPS tags
                    self.set_attr_from_xmp_tag('longitude', xtags, [
                        '@drone-dji:Longitude'
                    ], float)
                    self.set_attr_from_xmp_tag('latitude', xtags, [
                        '@drone-dji:Latitude'
                    ], float)
                    self.set_attr_from_xmp_tag('altitude', xtags, [
                        '@drone-dji:AbsoluteAltitude'
                    ], float)

                    # Phantom 4 RTK
                    if '@drone-dji:RtkStdLon' in xtags:
                        y = float(self.get_xmp_tag(xtags, '@drone-dji:RtkStdLon'))
                        x = float(self.get_xmp_tag(xtags, '@drone-dji:RtkStdLat'))
                        self.gps_xy_stddev = max(x, y)

                        if '@drone-dji:RtkStdHgt' in xtags:
                            self.gps_z_stddev = float(self.get_xmp_tag(xtags, '@drone-dji:RtkStdHgt'))
                    else:
                        self.set_attr_from_xmp_tag('gps_xy_stddev', xtags, [
                            '@Camera:GPSXYAccuracy',
                            'GPSXYAccuracy'
                        ], float)
                        self.set_attr_from_xmp_tag('gps_z_stddev', xtags, [
                            '@Camera:GPSZAccuracy',
                            'GPSZAccuracy'
                        ], float)

                    # DJI Speed tags
                    if '@drone-dji:FlightXSpeed' in xtags and \
                       '@drone-dji:FlightYSpeed' in xtags and \
                       '@drone-dji:FlightZSpeed' in xtags:
                        self.set_attr_from_xmp_tag('speed_x', xtags, [
                            '@drone-dji:FlightXSpeed'
                        ], float)
                        self.set_attr_from_xmp_tag('speed_y', xtags, [
                            '@drone-dji:FlightYSpeed',
                        ], float)
                        self.set_attr_from_xmp_tag('speed_z', xtags, [
                            '@drone-dji:FlightZSpeed',
                        ], float)

                    # Account for over-estimation
                    if self.gps_xy_stddev is not None:
                        self.gps_xy_stddev *= 2.0
                    if self.gps_z_stddev is not None:
                        self.gps_z_stddev *= 2.0

                    if 'DLS:Yaw' in xtags:
                        self.set_attr_from_xmp_tag('dls_yaw', xtags, ['DLS:Yaw'], float)
                        self.set_attr_from_xmp_tag('dls_pitch', xtags, ['DLS:Pitch'], float)
                        self.set_attr_from_xmp_tag('dls_roll', xtags, ['DLS:Roll'], float)

                    camera_projection = self.get_xmp_tag(xtags, ['@Camera:ModelType', 'Camera:ModelType'])
                    if camera_projection is not None:
                        camera_projection = camera_projection.lower()
                        if camera_projection in projections:
                            self.camera_projection = camera_projection

                    # OPK
                    self.set_attr_from_xmp_tag('yaw', xtags, ['@drone-dji:FlightYawDegree', '@Camera:Yaw', 'Camera:Yaw', 'Camera:IrradianceYaw'], float)
                    self.set_attr_from_xmp_tag('pitch', xtags, ['@drone-dji:GimbalPitchDegree', '@Camera:Pitch', 'Camera:Pitch', 'Camera:IrradiancePitch'], float)
                    self.set_attr_from_xmp_tag('roll', xtags, ['@drone-dji:GimbalRollDegree', '@Camera:Roll', 'Camera:Roll', 'Camera:IrradianceRoll'], float)

                    # Normalize YPR conventions (assuming nadir camera)
                    # Yaw: 0 --> top of image points north
                    # Yaw: 90 --> top of image points east
                    # Yaw: 270 --> top of image points west
                    # Pitch: 0 --> nadir camera
                    # Pitch: 90 --> camera is looking forward
                    # Roll: 0 (assuming gimbal)
                    if self.has_ypr():
                        if self.camera_make.lower() in ['dji', 'hasselblad']:
                            self.pitch = 90 + self.pitch

                        if self.camera_make.lower() == 'sensefly':
                            self.roll *= -1

                except Exception as e:
                    log.ODM_WARNING("Cannot read XMP tags for %s: %s" % (self.filename, str(e)))

                # self.set_attr_from_xmp_tag('center_wavelength', xtags, [
                #     'Camera:CentralWavelength'
                # ], float)

                # self.set_attr_from_xmp_tag('bandwidth', xtags, [
                #     'Camera:WavelengthFWHM'
                # ], float)

        # Special case band handling for AeroVironment Quantix images
        # for some reason, they don't store band information in EXIFs
        if self.camera_make.lower() == 'aerovironment' and \
            self.camera_model.lower() == 'quantix':
            matches = re.match("IMG_(\d+)_(\w+)\.\w+", self.filename, re.IGNORECASE)
            if matches:
                band_aliases = {
                    'GRN': 'Green',
                    'NIR': 'Nir',
                    'RED': 'Red',
                    'RGB': 'RedGreenBlue',
                }
                self.capture_uuid = matches.group(1)
                self.band_name = band_aliases.get(matches.group(2), matches.group(2))

        # Sanitize band name since we use it in folder paths
        self.band_name = re.sub('[^A-Za-z0-9]+', '', self.band_name)

        self.compute_focal(tags, xtags)
        self.compute_opk()

    def compute_focal(self, tags, xtags):
        try:
            _focal_ratio, _focal_length = self.extract_focal(self.camera_make, self.camera_model, tags, xtags)
            self.focal_ratio = _focal_ratio
            if self.focal_length is None:
                self.focal_length = _focal_length
        except (IndexError, ValueError) as e:
            log.ODM_WARNING("Cannot extract focal ratio for %s: %s" % (self.filename, str(e)))

    def extract_focal(self, make, model, tags, xtags):
        if make != "unknown":
            # remove duplicate 'make' information in 'model'
            model = model.replace(make, "")

        sensor_string = (make.strip() + " " + model.strip()).strip().lower()

        sensor_width = None
        if ("EXIF FocalPlaneResolutionUnit" in tags and "EXIF FocalPlaneXResolution" in tags):
            resolution_unit = self.float_value(tags["EXIF FocalPlaneResolutionUnit"])
            mm_per_unit = get_mm_per_unit(resolution_unit)
            if mm_per_unit:
                pixels_per_unit = self.float_value(tags["EXIF FocalPlaneXResolution"])
                if pixels_per_unit <= 0 and "EXIF FocalPlaneYResolution" in tags:
                    pixels_per_unit = self.float_value(tags["EXIF FocalPlaneYResolution"])

                if pixels_per_unit > 0 and self.width is not None:
                    units_per_pixel = 1 / pixels_per_unit
                    sensor_width = self.width * units_per_pixel * mm_per_unit

        focal_35 = None
        focal = None
        if "EXIF FocalLengthIn35mmFilm" in tags:
            focal_35 = self.float_value(tags["EXIF FocalLengthIn35mmFilm"])
        if "EXIF FocalLength" in tags:
            focal = self.float_value(tags["EXIF FocalLength"])
        if focal is None and "@aux:Lens" in xtags:
            lens = self.get_xmp_tag(xtags, ["@aux:Lens"])
            matches = re.search('([\d\.]+)mm', str(lens))
            if matches:
                focal = float(matches.group(1))

        if focal_35 is not None and focal_35 > 0:
            focal_ratio = focal_35 / 36.0  # 35mm film produces 36x24mm pictures.
        else:
            if not sensor_width:
                sensor_width = sensor_data().get(sensor_string, None)
            if sensor_width and focal:
                focal_ratio = focal / sensor_width
            else:
                focal_ratio = 0.85

        return focal_ratio, focal

    def set_attr_from_xmp_tag(self, attr, xmp_tags, tags, cast=None):
        v = self.get_xmp_tag(xmp_tags, tags)
        if v is not None:
            if cast is None:
                setattr(self, attr, v)
            else:
                # Handle fractions
                if (cast == float or cast == int) and "/" in v:
                    v = self.try_parse_fraction(v)
                setattr(self, attr, cast(v))

    def get_xmp_tag(self, xmp_tags, tags):
        if isinstance(tags, str):
            tags = [tags]

        for tag in tags:
            if tag in xmp_tags:
                t = xmp_tags[tag]

                if isinstance(t, string_types):
                    return str(t)
                elif isinstance(t, dict):
                    items = t.get('rdf:Seq', {}).get('rdf:li', {})
                    if items:
                        if isinstance(items, string_types):
                            return items
                        return " ".join(items)
                elif isinstance(t, int) or isinstance(t, float):
                    return t


    # From https://github.com/mapillary/OpenSfM/blob/master/opensfm/exif.py
    def get_xmp(self, file):
        img_bytes = file.read()
        xmp_start = img_bytes.find(b'<x:xmpmeta')
        xmp_end = img_bytes.find(b'</x:xmpmeta')

        if xmp_start < xmp_end:
            xmp_str = img_bytes[xmp_start:xmp_end + 12].decode('utf8')
            try:
                xdict = x2d.parse(xmp_str)
            except ExpatError as e:
                from bs4 import BeautifulSoup
                xmp_str = str(BeautifulSoup(xmp_str, 'xml'))
                xdict = x2d.parse(xmp_str)
                log.ODM_WARNING("%s has malformed XMP XML (but we fixed it)" % self.filename)
            xdict = xdict.get('x:xmpmeta', {})
            xdict = xdict.get('rdf:RDF', {})
            xdict = xdict.get('rdf:Description', {})
            if isinstance(xdict, list):
                return xdict
            else:
                return [xdict]
        else:
            return []

    def dms_to_decimal(self, dms, sign):
        """Converts dms coords to decimal degrees"""
        degrees, minutes, seconds = self.float_values(dms)

        if degrees is not None and minutes is not None and seconds is not None:
            return (-1 if sign.values[0] in 'SWsw' else 1) * (
                degrees +
                minutes / 60 +
                seconds / 3600
            )

    def float_values(self, tag):
        if isinstance(tag.values, list):
            result = []
            for v in tag.values:
                if isinstance(v, int):
                    result.append(float(v))
                elif isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], float):
                    result.append(v[0])
                elif v.den != 0:
                    result.append(float(v.num) / float(v.den))
                else:
                    result.append(None)
            return result
        else:
            return [float(tag.values.num) / float(tag.values.den) if tag.values.den != 0 else None]

    def float_value(self, tag):
        v = self.float_values(tag)
        if len(v) > 0:
            return v[0]

    def int_values(self, tag):
        if isinstance(tag.values, list):
            return [int(v) for v in tag.values]
        else:
            return [int(tag.values)]

    def int_value(self, tag):
        v = self.int_values(tag)
        if len(v) > 0:
            return v[0]

    def list_values(self, tag):
        return " ".join(map(str, tag.values))

    def try_parse_fraction(self, val):
        parts = val.split("/")
        if len(parts) == 2:
            try:
                num, den = map(float, parts)
                return num / den if den != 0 else val
            except ValueError:
                pass
        return val

    def get_size(self):
        return self.width, self.height

    def get_radiometric_calibration(self):
        if self.camera_make == "Parrot" and self.camera_model == "Sequoia":
            return self.get_radiometric_calibration_sequoia()

        if isinstance(self.radiometric_calibration, str):
            parts = self.radiometric_calibration.split(" ")
            if len(parts) == 3:
                return list(map(float, parts))

        return [None, None, None]

    def get_dark_level(self):
        if self.camera_make == "Parrot" and self.camera_model == "Sequoia":
            return self.get_dark_level_sequoia()

        if self.black_level:
            levels = np.array([float(v) for v in self.black_level.split(" ")])
            return levels.mean()

    def get_gain(self):
        if self.gain is not None:
            return self.gain
        elif self.iso_speed:
            #(gain = ISO/100)
            return self.iso_speed / 100.0

    def get_vignetting_center(self):
        if self.vignetting_center:
            parts = self.vignetting_center.split(" ")
            if len(parts) == 2:
                return list(map(float, parts))
        return [None, None]

    def get_vignetting_polynomial(self):
        if self.vignetting_polynomial:
            parts = self.vignetting_polynomial.split(" ")
            if len(parts) > 0:
                coeffs = list(map(float, parts))

                # Different camera vendors seem to use different ordering for the coefficients
                if self.camera_make != "Sentera":
                    coeffs.reverse()

                return coeffs

    def get_principal_point(self):
        if self.principal_point is not None:
            parts = self.principal_point.split(",")
            if len(parts) > 0:
                return list(map(float, parts))
        return None

    def get_distortion_parameters(self):
        if self.distortion_parameters is not None:
            parts = self.distortion_parameters.split(" ")
            if len(parts) > 0:
                return list(map(float, parts))
        return None

    def get_focal_plane_resolution(self):
        if self.focal_plane_resolution_x is not None and self.focal_plane_resolution_y is not None:
            return self.focal_plane_resolution_x, self.focal_plane_resolution_y
        return None

    def get_utc_time(self):
        if self.utc_time:
            return datetime.fromtimestamp(self.utc_time / 1000, timezone.utc)

    def get_photometric_exposure(self):
        # H ~= (exposure_time) / (f_number^2)
        if self.fnumber is not None and self.exposure_time is not None and self.exposure_time > 0 and self.fnumber > 0:
            return self.exposure_time / (self.fnumber * self.fnumber)

    def get_horizontal_irradiance(self):
        scale = self.get_irradiance_scale()
        if self.horizontal_irradiance is not None:
            return self.horizontal_irradiance * scale
        elif self.spectral_irradiance is not None:
            return self.spectral_irradiance * scale
        else:
            return None

    def get_sun_sensor(self):
        if self.camera_make == "Parrot" and self.camera_model == "Sequoia":
            return self.get_sun_sensor_sequoia()

        if self.sun_sensor is not None:
            # TODO: Presence of XMP:SunSensorExposureTime
            # and XMP:SunSensorSensitivity might
            # require additional logic. If these two tags are present,
            # then sun_sensor is not in physical units?
            return self.sun_sensor / 65535.0 # normalize uint16 (assumed)
        elif self.spectral_irradiance is not None:
            return self.spectral_irradiance * self.get_irradiance_scale()
        else:
            return None

    def get_irradiance_scale(self):
        scale = 1.0
        if self.irradiance_scale_to_si is not None:
            scale = self.irradiance_scale_to_si
        elif self.camera_make == 'MicaSense':
            # In the case of Micasense DLS2 missing the scale metadata,
            # multiply the SpectralIrradiance and HorizontalIrradiance by 0.01 to get to W/m2/nm.
            # See https://github.com/micasense/imageprocessing/issues/47
            scale = 0.01
        elif self.camera_make == 'DJI':
            # Phantom 4 Multispectral saves irridiance in @drone-dji:Irradiance and Camera:Irradiance
            scale = 1.0 / math.pi

        return scale

    def get_dls_pose(self):
        if self.dls_yaw is not None:
            return [self.dls_yaw, self.dls_pitch, self.dls_roll]
        return [0.0, 0.0, 0.0]

    def get_bit_depth_max(self):
        if self.bits_per_sample:
            return float(2 ** self.bits_per_sample)
        else:
            # If it's a JPEG, this must be 256
            _, ext = os.path.splitext(self.filename)
            if ext.lower() in [".jpeg", ".jpg"]:
                return 256.0

        return None

    def get_rig_relatives(self):
        if self.rig_relatives is not None:
            parts = self.rig_relatives.split(",")
            if len(parts) > 0:
                return list(map(float, parts))
        return None

    def get_capture_id(self):
        # Use capture UUID first, capture time as fallback
        if self.capture_uuid is not None:
            return self.capture_uuid

        return self.get_utc_time()

    def get_gps_dop(self):
        val = -9999
        if self.gps_xy_stddev is not None:
            val = self.gps_xy_stddev
        if self.gps_z_stddev is not None:
            val = max(val, self.gps_z_stddev)
        if val > 0:
            return val

        return None

    def override_gps_dop(self, dop):
        self.gps_xy_stddev = self.gps_z_stddev = dop

    def override_camera_projection(self, camera_projection):
        if camera_projection in projections:
            self.camera_projection = camera_projection

    def is_thermal(self):
        #Added for support M2EA camera sensor
        if(self.camera_make == "DJI" and self.camera_model == "MAVIC2-ENTERPRISE-ADVANCED" and self.width == 640 and self.height == 512):
            return True
        #Added for support DJI H20T camera sensor
        if(self.camera_make == "DJI" and self.camera_model == "ZH20T" and self.width == 640 and self.height == 512):
            return True
        return self.band_name.upper() in ["LWIR"] # TODO: more?

    def is_rgb(self):
        return self.band_name.upper() in ["RGB", "REDGREENBLUE"]

    def camera_id(self):
        return " ".join(
                [
                    "v2",
                    self.camera_make.strip(),
                    self.camera_model.strip(),
                    str(int(self.width)),
                    str(int(self.height)),
                    self.camera_projection,
                    str(float(self.focal_ratio))[:6],
                ]
            ).lower()

    def to_opensfm_exif(self, rolling_shutter = False, rolling_shutter_readout = 0):
        capture_time = 0.0
        if self.utc_time is not None:
            capture_time = self.utc_time / 1000.0

        gps = {}
        has_gps = self.latitude is not None and self.longitude is not None
        if has_gps:
            gps['latitude'] = self.latitude
            gps['longitude'] = self.longitude
            if self.altitude is not None:
                gps['altitude'] = self.altitude
            else:
                gps['altitude'] = 0.0

            dop = self.get_gps_dop()
            if dop is None:
                dop = 10.0 # Default

            gps['dop'] = dop

        d = {
            "make": self.camera_make,
            "model": self.camera_model,
            "width": self.width,
            "height": self.height,
            "projection_type": self.camera_projection,
            "focal_ratio": self.focal_ratio,
            "orientation": self.orientation,
            "capture_time": capture_time,
            "gps": gps,
            "camera": self.camera_id()
        }

        if self.has_opk():
            d['opk'] = {
                'omega': self.omega,
                'phi': self.phi,
                'kappa': self.kappa
            }

        # Speed is not useful without GPS
        if self.has_speed() and has_gps:
            d['speed'] = [self.speed_y, self.speed_x, self.speed_z]

        if rolling_shutter:
            d['rolling_shutter'] = get_rolling_shutter_readout(self, rolling_shutter_readout)

        return d

    def has_ypr(self):
        return self.yaw is not None and \
            self.pitch is not None and \
            self.roll is not None

    def has_opk(self):
        return self.omega is not None and \
            self.phi is not None and \
            self.kappa is not None

    def has_speed(self):
        return self.speed_x is not None and \
                self.speed_y is not None and \
                self.speed_z is not None

    def has_geo(self):
        return self.latitude is not None and \
            self.longitude is not None

    def compute_opk(self):
        if self.has_ypr() and self.has_geo():
            y, p, r = math.radians(self.yaw), math.radians(self.pitch), math.radians(self.roll)

            # Ref: New Calibration and Computing Method for Direct
            # Georeferencing of Image and Scanner Data Using the
            # Position and Angular Data of an Hybrid Inertial Navigation System
            # by Manfred Bäumker

            # YPR rotation matrix
            cnb = np.array([[ math.cos(y) * math.cos(p), math.cos(y) * math.sin(p) * math.sin(r) - math.sin(y) * math.cos(r), math.cos(y) * math.sin(p) * math.cos(r) + math.sin(y) * math.sin(r)],
                            [ math.sin(y) * math.cos(p), math.sin(y) * math.sin(p) * math.sin(r) + math.cos(y) * math.cos(r), math.sin(y) * math.sin(p) * math.cos(r) - math.cos(y) * math.sin(r)],
                            [ -math.sin(p), math.cos(p) * math.sin(r), math.cos(p) * math.cos(r)],
                           ])

            # Convert between image and body coordinates
            # Top of image pixels point to flying direction
            # and camera is looking down.
            # We might need to change this if we want different
            # camera mount orientations (e.g. backward or sideways)

            # (Swap X/Y, flip Z)
            cbb = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])

            delta = 1e-7

            alt = self.altitude if self.altitude is not None else 0.0
            p1 = np.array(ecef_from_lla(self.latitude + delta, self.longitude, alt))
            p2 = np.array(ecef_from_lla(self.latitude - delta, self.longitude, alt))
            xnp = p1 - p2
            m = np.linalg.norm(xnp)

            if m == 0:
                log.ODM_WARNING("Cannot compute OPK angles, divider = 0")
                return

            # Unit vector pointing north
            xnp /= m

            znp = np.array([0, 0, -1]).T
            ynp = np.cross(znp, xnp)

            cen = np.array([xnp, ynp, znp]).T

            # OPK rotation matrix
            ceb = cen.dot(cnb).dot(cbb)

            self.omega = math.degrees(math.atan2(-ceb[1][2], ceb[2][2]))
            self.phi = math.degrees(math.asin(ceb[0][2]))
            self.kappa = math.degrees(math.atan2(-ceb[0][1], ceb[0][0]))

    def get_capture_megapixels(self):
        if self.exif_width is not None and self.exif_height is not None:
            # Accurate so long as resizing / postprocess software
            # did not fiddle with the tags
            return self.exif_width * self.exif_height / 1e6
        elif self.width is not None and self.height is not None:
            # Fallback, might not be accurate since the image
            # could have been resized
            return self.width * self.height / 1e6
        else:
            return 0.0

    ######################################################################################################################
    # Adapting MicaSense image processing script for thermal band alignment
    # - https://github.com/micasense/imageprocessing/blob/236cd23978116fd4cdfc79e4cd1cdc68e1b004d8/micasense/imageutils.py
    # - https://github.com/micasense/imageprocessing/blob/236cd23978116fd4cdfc79e4cd1cdc68e1b004d8/micasense/image.py
    ######################################################################################################################
    def principal_point_px(self):
        principal_point = self.get_principal_point()
        focal_plane_resolution = self.get_focal_plane_resolution()
        center_x = principal_point[0] * focal_plane_resolution[0]
        center_y = principal_point[1] * focal_plane_resolution[1]
        return (center_x, center_y)

    def cv2_camera_matrix(self):
        center_x, center_y = self.principal_point_px()
        focal_plane_resolution = self.get_focal_plane_resolution()

        # set up camera matrix for cv2
        cam_mat = np.zeros((3, 3))
        cam_mat[0, 0] = self.focal_length * focal_plane_resolution[0]
        cam_mat[1, 1] = self.focal_length * focal_plane_resolution[1]
        cam_mat[2, 2] = 1.0
        cam_mat[0, 2] = center_x
        cam_mat[1, 2] = center_y

        return cam_mat

    def cv2_distortion_coeff(self):
        return np.array(self.get_distortion_parameters())[[0, 1, 3, 4, 2]]

    def get_homography(self, ref, R=None, T=None, undistorted=True):
        ''' get the homography that maps from this image to the reference image '''

        # convert euler angles to a rotation matrix
        def rotations_degrees_to_rotation_matrix(rotation_degrees):
            cx = np.cos(np.deg2rad(rotation_degrees[0]))
            cy = np.cos(np.deg2rad(rotation_degrees[1]))
            cz = np.cos(np.deg2rad(rotation_degrees[2]))
            sx = np.sin(np.deg2rad(rotation_degrees[0]))
            sy = np.sin(np.deg2rad(rotation_degrees[1]))
            sz = np.sin(np.deg2rad(rotation_degrees[2]))

            Rx = np.mat([  1,  0,  0,
                            0, cx,-sx,
                            0, sx, cx]).reshape(3,3)
            Ry = np.mat([ cy,  0, sy,
                            0,  1,  0,
                            -sy,  0, cy]).reshape(3,3)
            Rz = np.mat([ cz,-sz,  0,
                            sz, cz,  0,
                            0,  0,  1]).reshape(3,3)
            R = Rx*Ry*Rz
            return R

        if R is None:
            R = rotations_degrees_to_rotation_matrix(self.get_rig_relatives())
        if T is None:
            T =np.zeros(3)
        R_ref = rotations_degrees_to_rotation_matrix(ref.get_rig_relatives())
        A = np.zeros((4,4))
        A[0:3,0:3]=np.dot(R_ref.T,R)
        A[0:3,3]=T
        A[3,3]=1.

        C, _ = cv2.getOptimalNewCameraMatrix(self.cv2_camera_matrix(),
                                            self.cv2_distortion_coeff() if undistorted else None,
                                            self.get_size(), alpha=1)
        Cr, _ = cv2.getOptimalNewCameraMatrix(ref.cv2_camera_matrix(),
                                            ref.cv2_distortion_coeff() if undistorted else None,
                                            ref.get_size(), alpha=1)
        CC = np.zeros((4,4))
        CC[0:3,0:3] = C
        CC[3,3]=1.
        CCr = np.zeros((4,4))
        CCr[0:3,0:3] = Cr
        CCr[3,3]=1.

        B = np.array(np.dot(CCr,np.dot(A,np.linalg.inv(CC))))
        B[:,2]=B[:,2]-B[:,3]
        B = B[0:3,0:3]
        B = B/B[2,2]
        return np.array(B)

    def undistorted(self, image):
        ''' return the undistorted image from input image '''
        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.cv2_camera_matrix(),
                                                       self.cv2_distortion_coeff(),
                                                       self.get_size(),
                                                       alpha=1)
        map1, map2 = cv2.initUndistortRectifyMap(self.cv2_camera_matrix(),
                                                self.cv2_distortion_coeff(),
                                                np.eye(3),
                                                new_cam_mat,
                                                self.get_size(),
                                                cv2.CV_32F)
        # compute the undistorted 16 bit image
        return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    ######################################################################################################################
    # Parrot Sequoia image handler
    # - https://github.com/dobedobedo/Parrot_Sequoia_Image_Handler
    # - https://github.com/OpenDroneMap/ODM/commit/70b2913ec0377e72591288af0500de7c153b3656
    ######################################################################################################################

    def get_dark_level_sequoia(self):
        if self.sensor_model_coefficient is None:       # Exif SensorModel is missing
            return None

        sensor_coefs = np.array([float(v.strip(".")) for v in self.sensor_model_coefficient.split(",")])
        return sensor_coefs[1]

    def get_radiometric_calibration_sequoia(self):
        if self.sensor_model_coefficient is None:       # Exif SensorModel is missing
            return [None, None, None]

        gain_scale = 100 * self.get_gain() * self.exposure_time
        sensor_coefs = np.array([float(v.strip(".")) for v in self.sensor_model_coefficient.split(",")])
        a1 = gain_scale * math.pow(self.fnumber, 2) / ((sensor_coefs[0] * self.exposure_time * self.iso_speed + sensor_coefs[2]) / 65536.0)
        return [a1, None, None]

    def get_sun_sensor_sequoia(self):
        if self.irradiance_calibration is None:      # Exif IrradianceCalibrationMeasurement is missing
            return None

        irrad_cal_pm = np.array([float(v) for v in self.irradiance_calibration.split(",")])
        #    GainIndex          IntegTime              CH0              CH1
        # irrad_cal_pm[0]=0  irrad_cal_pm[1]  irrad_cal_pm[2]  irrad_cal_pm[3]
        # irrad_cal_pm[4]=1  irrad_cal_pm[5]  irrad_cal_pm[6]  irrad_cal_pm[7]
        # irrad_cal_pm[8]=2  irrad_cal_pm[9]  irrad_cal_pm[10] irrad_cal_pm[11]
        # irrad_cal_pm[12]=3 irrad_cal_pm[13] irrad_cal_pm[14] irrad_cal_pm[15]
        irrad_cal_0 = irrad_cal_pm[2] * (600.0 / irrad_cal_pm[1])
        irrad_cal_1 = irrad_cal_pm[6] * (600.0 / irrad_cal_pm[5])
        irrad_cal_2 = irrad_cal_pm[10] * (600.0 / irrad_cal_pm[9])
        irrad_cal_3 = irrad_cal_pm[14] * (600.0 / irrad_cal_pm[13])
        irrad_cal_fac = [irrad_cal_0, irrad_cal_1, irrad_cal_2, irrad_cal_3]

        irrad_binary = base64.b64decode(self.irradiance_list)

        irrad_last3 = None
        irrad_last2 = None
        irrad_last1 = None

        for irrad_pm in struct.iter_unpack("QHHHHfff", irrad_binary):
            #                               Q=uint64 H=uint16 f=float32
            # irrad_pm[0]=TimeStamp     irrad_pm[1]=CH0 count   irrad_pm[2]=CH1 count
            # irrad_pm[3]=GainIndex     irrad_pm[4]=IntegTime
            # irrad_pm[5]=Yaw           irrad_pm[6]=Pitch       irrad_pm[7]=Roll
            #
            irrad_fac = irrad_cal_fac[1]/irrad_cal_fac[irrad_pm[3]] * 600.0/irrad_pm[4]
            irrad_last3 = irrad_last2
            irrad_last2 = irrad_last1
            irrad_last1 = irrad_pm[1] * irrad_fac

        if irrad_last3 is None:     # No. of records less than 3
            return None
        else:
            irrad_avg = (irrad_last1 + irrad_last2 + irrad_last3) / 3.0
            return irrad_avg / irrad_cal_1
