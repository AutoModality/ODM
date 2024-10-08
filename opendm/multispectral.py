import math
from pickletools import optimize
import re
import cv2
import os
from opendm import dls
import numpy as np
from opendm import log
from opendm.concurrency import parallel_map
from opendm import thermal
from opensfm.io import imread

from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank, gaussian
from skimage.util import img_as_ubyte

# Loosely based on https://github.com/micasense/imageprocessing/blob/master/micasense/utils.py

def dn_to_radiance(photo, image, band_vignetting_coefficients=None):
    """
    Convert Digital Number values to Radiance values
    :param photo ODM_Photo
    :param image numpy array containing image data
    :return numpy array with radiance image values
    """

    image = image.astype("float32")
    if len(image.shape) != 3:
        raise ValueError("Image should have shape length of 3 (got: %s)" % len(image.shape))

    # Thermal (this should never happen, but just in case..)
    if photo.is_thermal():
        return image

    # All others
    a1, a2, a3 = photo.get_radiometric_calibration()
    dark_level = photo.get_dark_level()

    exposure_time = photo.exposure_time
    gain = photo.get_gain()
    gain_adjustment = photo.gain_adjustment
    photometric_exp = photo.get_photometric_exposure()

    if a1 is None and photometric_exp is None:
        log.ODM_WARNING("Cannot perform radiometric calibration, no FNumber/Exposure Time or Radiometric Calibration EXIF tags found in %s. Using Digital Number." % photo.filename)
        return image

    if a1 is None and photometric_exp is not None:
        a1 = photometric_exp

    if band_vignetting_coefficients is not None:
        V = band_vignetting_coefficients
        x, y = np.meshgrid(np.arange(photo.width), np.arange(photo.height))
    else:
        V, x, y = vignette_map(photo)
        if V is not None:
            V = np.repeat(V[:, :, np.newaxis], image.shape[2], axis=2)
        if x is None:
            x, y = np.meshgrid(np.arange(photo.width), np.arange(photo.height))

    if dark_level is not None:
        image -= dark_level

    # Normalize DN to 0 - 1.0
    bit_depth_max = photo.get_bit_depth_max()
    if bit_depth_max:
        image /= bit_depth_max
    else:
        log.ODM_WARNING("Cannot normalize DN for %s, bit depth is missing" % photo.filename)

    # Vignette correction
    if V is not None:
        image *= V

    # Row gradient correction
    if exposure_time and a2 is not None and a3 is not None:
        R = 1.0 / (1.0 + a2 * y / exposure_time - a3 * y)
        R = np.repeat(R[:, :, np.newaxis], image.shape[2], axis=2)
        image *= R

    # Floor any negative radiances to zero (can happen due to noise around blackLevel)
    if dark_level is not None:
        image[image < 0] = 0

    # Apply the radiometric calibration - i.e. scale by the gain-exposure product and
    # multiply with the radiometric calibration coefficient

    if gain is not None and exposure_time is not None:
        image /= (gain * exposure_time)

    if photo.camera_make == "DJI" and gain_adjustment is not None:
        image *= gain_adjustment
    else:
        image *= a1

    return image

def vignette_map(photo):
    x_vc, y_vc = photo.get_vignetting_center()
    polynomial = photo.get_vignetting_polynomial()

    if x_vc and polynomial:
        # append 1., so that we can call with numpy polyval
        polynomial.append(1.0)
        vignette_poly = np.array(polynomial)

        # perform vignette correction
        # get coordinate grid across image
        x, y = np.meshgrid(np.arange(photo.width), np.arange(photo.height))

        # meshgrid returns transposed arrays
        # x = x.T
        # y = y.T

        # compute matrix of distances from image center
        r = np.hypot((x - x_vc), (y - y_vc))

        # compute the vignette polynomial for each distance - we divide by the polynomial so that the
        # corrected image is image_corrected = image_original * vignetteCorrection
        vignette = np.polyval(vignette_poly, r)

        # DJI is special apparently
        if photo.camera_make != "DJI":
            vignette = 1.0 / vignette

        return vignette, x, y

    return None, None, None

def dn_to_reflectance(photo, image, band_irradiance, band_vignetting=None, use_sun_sensor=True):
    radiance = dn_to_radiance(photo, image, band_vignetting)

    if band_irradiance is not None and use_sun_sensor is not True:
        irradiance = band_irradiance
    else:
        irradiance = compute_irradiance(photo, use_sun_sensor=use_sun_sensor)

    reflectance = radiance * math.pi / irradiance

    return reflectance

def compute_irradiance(photo, use_sun_sensor=True):
    # Thermal (this should never happen, but just in case..)
    if photo.is_thermal():
        return 1.0

    # Some cameras (e.g. Micasense and DJI) store the value in metadata
    hirradiance = photo.get_horizontal_irradiance()
    if hirradiance is not None:
        return hirradiance

    # TODO: support for calibration panels

    # For general cases
    if use_sun_sensor and photo.get_sun_sensor():
        # Estimate it
        dls_orientation_vector = np.array([0,0,-1])
        sun_vector_ned, sensor_vector_ned, sun_sensor_angle, \
        solar_elevation, solar_azimuth = dls.compute_sun_angle([photo.latitude, photo.longitude],
                                        photo.get_dls_pose(),
                                        photo.get_utc_time(),
                                        dls_orientation_vector)

        angular_correction = dls.fresnel(sun_sensor_angle)

        # TODO: support for direct and scattered irradiance

        direct_to_diffuse_ratio = 6.0 # Assumption, clear skies
        spectral_irradiance = photo.get_sun_sensor()

        percent_diffuse = 1.0 / direct_to_diffuse_ratio
        sensor_irradiance = spectral_irradiance / angular_correction

        # Find direct irradiance in the plane normal to the sun
        untilted_direct_irr = sensor_irradiance / (percent_diffuse + np.cos(sun_sensor_angle))
        direct_irradiance = untilted_direct_irr
        scattered_irradiance = untilted_direct_irr * percent_diffuse

        # compute irradiance on the ground using the solar altitude angle
        horizontal_irradiance = direct_irradiance * np.sin(solar_elevation) + scattered_irradiance
        return horizontal_irradiance
    elif use_sun_sensor:
        log.ODM_WARNING("No sun sensor values found for %s" % photo.filename)

    return 1.0

def radiometric_calibrate(photo, image, image_type='reflectance', irradiance_by_hand=None, vignetting_info=None, use_sun_sensor=True):
    band_irradiance_mean = None
    if irradiance_by_hand is not None:
        band_irradiance_mean = irradiance_by_hand.get(photo.band_name)

    band_vignette_map = None
    if vignetting_info is not None:
        band_vignette_map = vignetting_info.get(photo.band_name)

    if not photo.is_thermal():
        if image_type == 'reflectance':
            return dn_to_reflectance(photo, image, band_irradiance_mean, band_vignette_map, use_sun_sensor)
        else:
            return dn_to_radiance(photo, image, band_vignette_map)
    else:
        return thermal.dn_to_temperature(photo, image)

def get_photos_by_band(multi_camera, user_band_name):
    band_name = get_primary_band_name(multi_camera, user_band_name)

    for band in multi_camera:
        if band['name'] == band_name:
            return band['photos']

def get_primary_band_name(multi_camera, user_band_name):
    if len(multi_camera) < 1:
        raise Exception("Invalid multi_camera list")

    # multi_camera is already sorted by band_index
    if user_band_name == "auto":
        return multi_camera[0]['name']

    for band in multi_camera:
        if band['name'].lower() == user_band_name.lower():
            return band['name']

    band_name_fallback = multi_camera[0]['name']

    log.ODM_WARNING("Cannot find band name \"%s\", will use \"%s\" instead" % (user_band_name, band_name_fallback))
    return band_name_fallback

def compute_band_maps(multi_camera, primary_band):
    """
    Computes maps of:
     - { photo filename --> associated primary band photo } (s2p)
     - { primary band filename --> list of associated secondary band photos } (p2s)
    by looking at capture UUID, capture time or filenames as a fallback
    """
    band_name = get_primary_band_name(multi_camera, primary_band)
    primary_band_photos = None
    for band in multi_camera:
        if band['name'] == band_name:
            primary_band_photos = band['photos']
            break

    # Try using capture time as the grouping factor
    try:
        unique_id_map = {}
        s2p = {}
        p2s = {}

        for p in primary_band_photos:
            uuid = p.get_capture_id()
            if uuid is None:
                raise Exception("Cannot use capture time (no information in %s)" % p.filename)

            # Should be unique across primary band
            if unique_id_map.get(uuid) is not None:
                raise Exception("Unreliable UUID/capture time detected (duplicate)")

            unique_id_map[uuid] = p

        for band in multi_camera:
            photos = band['photos']

            for p in photos:
                uuid = p.get_capture_id()
                if uuid is None:
                    raise Exception("Cannot use UUID/capture time (no information in %s)" % p.filename)

                # Should match the primary band
                if unique_id_map.get(uuid) is None:
                    raise Exception("Unreliable UUID/capture time detected (no primary band match)")

                s2p[p.filename] = unique_id_map[uuid]

                if band['name'] != band_name:
                    p2s.setdefault(unique_id_map[uuid].filename, []).append(p)

                # log.ODM_INFO("File %s <-- Capture %s" % (p.filename, uuid))

        return s2p, p2s
    except Exception as e:
        # Fallback on filename conventions
        log.ODM_WARNING("%s, will use filenames instead" % str(e))

        filename_map = {}
        s2p = {}
        p2s = {}
        file_regex = re.compile(r"^(.+)[-_]\w+(\.[A-Za-z]{3,4})$")

        for p in primary_band_photos:
            filename_without_band = re.sub(file_regex, "\\1\\2", p.filename)

            # Quick check
            if filename_without_band == p.filename:
                raise Exception("Cannot match bands by filename on %s, make sure to name your files [filename]_band[.ext] uniformly." % p.filename)

            filename_map[filename_without_band] = p

        for band in multi_camera:
            photos = band['photos']

            for p in photos:
                filename_without_band = re.sub(file_regex, "\\1\\2", p.filename)

                # Quick check
                if filename_without_band == p.filename:
                    raise Exception("Cannot match bands by filename on %s, make sure to name your files [filename]_band[.ext] uniformly." % p.filename)

                s2p[p.filename] = filename_map[filename_without_band]

                if band['name'] != band_name:
                    p2s.setdefault(filename_map[filename_without_band].filename, []).append(p)

        return s2p, p2s

def compute_band_irradiances(multi_camera):
    log.ODM_INFO("Computing band irradiance")

    band_irradiance_info = {}
    for band in multi_camera:
        irradiances = []
        band_irradiance_mean = 1.0
        for p in get_photos_by_band(multi_camera, band['name']):
            hirradiance = p.get_horizontal_irradiance()
            if hirradiance is not None:
                irradiances.append(hirradiance)
        if len(irradiances) > 0:
            band_irradiance_mean = sum(irradiances) / len(irradiances)

        band_irradiance_info[band['name']] = band_irradiance_mean
        log.ODM_INFO("%s band's mean irradiance: %s" % (band['name'], band_irradiance_mean))

    return band_irradiance_info

def compute_alignment_matrices(multi_camera, primary_band_name, images_path, s2p, p2s, max_concurrency=1, max_samples=30,
                               irradiance_by_hand=None, use_sun_sensor=True, rig_optimization=False, use_local_homography=False):
    log.ODM_INFO("Computing band alignment")

    alignment_info = {}

    # For each secondary band
    for band in multi_camera:
        if band['name'] != primary_band_name:
            matrices_samples = []
            use_local_warp_matrix = use_local_homography # and band['name'] == 'LWIR'
            max_samples = max_samples if not use_local_warp_matrix and max_samples > 0 and max_samples < len(band['photos']) else len(band['photos'])

            def parallel_compute_homography(photo):
                filename = photo.filename
                try:
                    # Caculate the best warp matrix using a few samples in favor of performance
                    if not use_local_warp_matrix and len(matrices_samples) >= max_samples:
                        # log.ODM_INFO("Got enough samples for %s (%s)" % (band['name'], max_samples))
                        return

                    # Find good matrix candidates for alignment
                    primary_band_photo = s2p.get(filename)
                    if primary_band_photo is None:
                        log.ODM_WARNING("Cannot find primary band photo for %s" % filename)
                        return

                    warp_matrix, dimension, algo = compute_homography(os.path.join(images_path, filename),
                                                                      os.path.join(images_path, primary_band_photo.filename),
                                                                      photo,
                                                                      primary_band_photo,
                                                                      irradiance_by_hand,
                                                                      use_sun_sensor,
                                                                      rig_optimization)

                    if warp_matrix is not None:
                        log.ODM_INFO("%s --> %s good match" % (filename, primary_band_photo.filename))

                        matrices_samples.append({
                            'filename': filename,
                            'align_filename': primary_band_photo.filename,
                            'warp_matrix': warp_matrix,
                            'eigvals': np.linalg.eigvals(warp_matrix),
                            'dimension': dimension,
                            'algo': algo
                        })
                    else:
                        log.ODM_INFO("%s --> %s cannot be matched" % (filename, primary_band_photo.filename))
                except Exception as e:
                    log.ODM_WARNING("Failed to compute homography for %s: %s" % (filename, str(e)))

            parallel_map(parallel_compute_homography, band['photos'], max_concurrency, single_thread_fallback=False)

            if use_local_warp_matrix or max_samples > 100:
                # Method 1 (faster): Find the matrix that has the most common eigvals among all matrices. That should be the "best" alignment.
                for m1 in matrices_samples:
                    acc = np.array([0.0,0.0,0.0])
                    e = m1['eigvals']

                    for m2 in matrices_samples:
                        acc += abs(e - m2['eigvals'])

                    m1['score'] = acc.sum()

            else:
                # Method 2 (slower): Find the matrix that has the most common projections
                for m1 in matrices_samples:
                    score = 0.0
                    image_size = m1['warp_matrix']

                    for m2 in matrices_samples:
                        image_raw = imread(os.path.join(images_path, m2['filename']), unchanged=True, anydepth=True)
                        photo_raw = next((p for p in band['photos'] if p.filename == m2['filename']), None)
                        image_raw = radiometric_calibrate(photo_raw, image_raw, 'radiance', irradiance_by_hand, None, use_sun_sensor)
                        if image_raw.shape[2] == 3:
                            image_gray = to_8bit(cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY))
                        else:
                            image_gray = to_8bit(image_raw[:,:,0])

                        image_proj1 = align_image(image_gray, m1['warp_matrix'], m1['dimension'])
                        image_proj2 = align_image(image_gray, m2['warp_matrix'], m2['dimension'])

                        margin = 0.1 # use 80% of image area to compare
                        h, w = image_proj1.shape
                        x1 = int(w * margin)
                        y1 = int(h * margin)
                        x2 = int(w * (1-margin))
                        y2 = int(h * (1-margin))
                        image_size = (x2-x1+1, y2-y1+1)

                        image_proj1_samples = image_proj1[y1:y2, x1:x2]
                        image_proj2_samples = image_proj2[y1:y2, x1:x2]
                        diff = abs(np.subtract(image_proj1_samples, image_proj2_samples))
                        score += np.sum(diff) / (w*h)

                    # log.ODM_INFO("Warp matrix: %s (score: %s, sample pixels: %s x %s)" % (m1, score, image_size[0], image_size[1]))
                    m1['score'] = score

            # Sort
            matrices_samples.sort(key=lambda x: x['score'], reverse=False)

            if len(matrices_samples) > 0:
                best_candidate = matrices_samples[0]

                # Alignment matrices for all shots
                matrices_all = []

                for photo in [{'filename': p.filename} for p in band['photos']]:
                    primary_band_photo = s2p.get(photo['filename'])
                    local_warp_matrix = next((item for item in matrices_samples if item['filename'] == photo['filename']), None) # matrices_samples is a list

                    if use_local_warp_matrix and local_warp_matrix is not None:
                        matrices_all.append(local_warp_matrix)
                    else:
                        matrices_all.append({
                            'filename': photo['filename'],
                            'align_filename': primary_band_photo.filename,
                            'warp_matrix': best_candidate['warp_matrix'],
                            'eigvals': best_candidate['eigvals'],
                            'dimension': best_candidate['dimension'],
                            'algo': best_candidate['algo']
                        })

                alignment_info[band['name']] = matrices_all

                if use_local_warp_matrix:
                    log.ODM_INFO("%s band will be aligned using local warp matrices %s" % (band['name'], matrices_all))
                else:
                    log.ODM_INFO("%s band will be aligned using global warp matrix %s (score: %s)" % (band['name'], best_candidate['warp_matrix'], best_candidate['score']))
            else:
                log.ODM_WARNING("Cannot find alignment matrix for band %s, The band might end up misaligned!" % band['name'])

    return alignment_info

def compute_homography(image_filename, align_image_filename, photo, align_photo, irradiance_by_hand=None, use_sun_sensor=True, rig_optimization=False):
    try:
        # Convert images to grayscale if needed
        image = imread(image_filename, unchanged=True, anydepth=True)
        image = radiometric_calibrate(photo, image, 'radiance', irradiance_by_hand, None, use_sun_sensor)
        if image.shape[2] == 3:
            image_gray = to_8bit(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            image_gray = to_8bit(image[:,:,0])

        max_dim = max(image_gray.shape)
        # if max_dim <= 320:
        #    log.ODM_WARNING("Small image for band alignment (%sx%s), this might be tough to compute." % (image_gray.shape[1], image_gray.shape[0]))

        align_image = imread(align_image_filename, unchanged=True, anydepth=True)
        align_image = radiometric_calibrate(align_photo, align_image, 'radiance', irradiance_by_hand, None, use_sun_sensor)
        if align_image.shape[2] == 3:
            align_image_gray = to_8bit(cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY))
        else:
            align_image_gray = to_8bit(align_image[:,:,0])

        def compute_using(algorithm):
            try:
                h = algorithm(image_gray, align_image_gray)
            except Exception as e:
                log.ODM_WARNING("Cannot compute homography: %s" % str(e))
                return None, (None, None)

            if h is None:
                return None, (None, None)

            det = np.linalg.det(h)

            # Check #1 homography's determinant will not be close to zero
            if abs(det) < 0.25:
                return None, (None, None)

            # Check #2 the ratio of the first-to-last singular value is sane (not too high)
            svd = np.linalg.svd(h, compute_uv=False)
            if svd[-1] == 0:
                return None, (None, None)

            ratio = svd[0] / svd[-1]
            if ratio > 100000:
                return None, (None, None)

            return h, (align_image_gray.shape[1], align_image_gray.shape[0])

        warp_matrix = None
        dimension = None
        algo = None

        if max_dim > 320:
            algo = 'feat'
            result = compute_using(find_features_homography)

            if result[0] is None:
                algo = 'ecc'
                log.ODM_INFO("Can't use features matching for %s, will use ECC (this might take a bit)" % photo.filename)
                result = compute_using(find_ecc_homography)

        else: # for low resolution images
            if photo.camera_make == 'MicaSense' and photo.band_name == 'LWIR':
                algo = 'rig'
                log.ODM_INFO("Using camera rig relatives to compute warp matrix for %s (rig relatives: %s)" % (photo.filename, str(photo.get_rig_relatives())))
                warp_matrix_intrinsic = find_rig_homography(photo, align_photo, image_gray, align_image_gray)
                if rig_optimization:
                    log.ODM_INFO("Using ECC to optimize the rig relatives warp matrix for %s" % photo.filename)
                    warp_matrix_ecc = find_ecc_homography(image_gray, align_image_gray, warp_matrix_init=warp_matrix_intrinsic)
                    if warp_matrix_ecc is not None:
                        algo = 'rig+ecc'
                        warp_matrix_optimized = np.array(np.dot(warp_matrix_ecc, warp_matrix_intrinsic))
                        warp_matrix_optimized /= warp_matrix_optimized[2,2]
                    else:
                        warp_matrix_optimized = warp_matrix_intrinsic
                        log.ODM_WARNING("Cannot compute ECC warp matrix for %s, use the rig relatives warp matrix instead" % photo.filename)

                else:
                    warp_matrix_optimized = warp_matrix_intrinsic
                result = warp_matrix_optimized, (align_image_gray.shape[1], align_image_gray.shape[0])

            else:
                algo = 'ecc'
                log.ODM_INFO("Using ECC for %s (this might take a bit)" % photo.filename)
                result = compute_using(find_ecc_homography)

        if result[0] is None:
            algo = None

        # log.ODM_INFO("Warp matrix for %s --> %s: \n%s (algorithm: %s)" % (photo.filename, align_photo.filename, str(result[0]), algo))

        warp_matrix, dimension = result
        return warp_matrix, dimension, algo

    except Exception as e:
        log.ODM_WARNING("Compute homography: %s" % str(e))
        return None, (None, None), None

def find_ecc_homography(image_gray, align_image_gray, number_of_iterations=2000, termination_eps=1e-8, start_eps=1e-4, warp_matrix_init=None):
    # Major props to Alexander Reynolds for his insight into the pyramided matching process found at
    # https://stackoverflow.com/questions/45997891/cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method
    pyramid_levels = 0
    h,w = image_gray.shape
    min_dim = min(h, w)

    if (min_dim <= 300):
        number_of_iterations = 5000
        termination_eps = 1e-6
        gaussian_filter_size = 9 # a constant since there is only one pyramid level
    else:
        gaussian_filter_size = 5 # will be increased in each pyramid level iteration

    while min_dim > 300:
        min_dim /= 2.0
        pyramid_levels += 1

    log.ODM_INFO("Pyramid levels: %s" % pyramid_levels)

    fx = align_image_gray.shape[1] / image_gray.shape[1]
    fy = align_image_gray.shape[0] / image_gray.shape[0]
    if warp_matrix_init is not None: # initial rough alignment
        image_gray = align_image(image_gray, warp_matrix_init, (align_image_gray.shape[1], align_image_gray.shape[0]),
                                 flags=(cv2.INTER_LINEAR if (fx < 1.0 and fy < 1.0) else cv2.INTER_CUBIC))
    else:
        if align_image_gray.shape[0] != image_gray.shape[0]:
            image_gray = cv2.resize(image_gray, None,
                                    fx=fx,
                                    fy=fy,
                                    interpolation=(cv2.INTER_AREA if (fx < 1.0 and fy < 1.0) else cv2.INTER_LANCZOS4))

    # Define the motion model, scale the initial warp matrix to smallest level
    default_matrix = np.eye(3, 3, dtype=np.float32)
    warp_matrix = default_matrix * np.array([[1,1,2],[1,1,2],[0.5,0.5,1]], dtype=np.float32)**(1-(pyramid_levels+1))

    # Build pyramids
    image_gray_pyr = [image_gray]
    align_image_pyr = [align_image_gray]

    for level in range(pyramid_levels):
        image_gray_pyr.insert(0, cv2.resize(image_gray_pyr[0], None, fx=1/2, fy=1/2,
                                interpolation=cv2.INTER_AREA))
        align_image_pyr.insert(0, cv2.resize(align_image_pyr[0], None, fx=1/2, fy=1/2,
                                interpolation=cv2.INTER_AREA))

    for level in range(pyramid_levels+1):
        ig = gradient(gaussian(normalize(image_gray_pyr[level])))
        aig = gradient(gaussian(normalize(align_image_pyr[level])))

        if level == pyramid_levels and pyramid_levels == 0:
            eps = termination_eps
        else:
            eps = start_eps - ((start_eps - termination_eps) / (pyramid_levels)) * level

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, eps)

        try:
            gaussian_filter_size = gaussian_filter_size + level * 2
            log.ODM_INFO("Computing ECC pyramid level %s using Gaussian filter size %s" % (level, gaussian_filter_size))
            _, warp_matrix = cv2.findTransformECC(ig, aig, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None, gaussFiltSize=gaussian_filter_size)
        except Exception as e:
            if level != pyramid_levels:
                log.ODM_INFO("Could not compute ECC warp_matrix at pyramid level %s, resetting matrix" % level)
                warp_matrix = default_matrix * np.array([[1,1,2],[1,1,2],[0.5,0.5,1]], dtype=np.float32)**(1-(pyramid_levels+1))
            else:
                # raise e
                return None


        if level != pyramid_levels:
            warp_matrix = warp_matrix * np.array([[1,1,2],[1,1,2],[0.5,0.5,1]], dtype=np.float32)

    return warp_matrix

def find_features_homography(image_gray, align_image_gray, feature_retention=0.8, min_match_count=4):

    # Detect SIFT features and compute descriptors.
    detector = cv2.SIFT_create() # edgeThreshold=10, contrastThreshold=0.1 (default 0.04)
    kp_image, desc_image = detector.detectAndCompute(image_gray, None)
    kp_align_image, desc_align_image = detector.detectAndCompute(align_image_gray, None)

    # Match
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(desc_image, desc_align_image, k=2)
    except Exception as e:
        return None

    # Filter good matches following Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < feature_retention * n.distance:
            good_matches.append(m)

    matches = good_matches

    if len(matches) < min_match_count:
        return None

    # Debug
    # imMatches = cv2.drawMatches(im1, kp_image, im2, kp_align_image, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)
    # log.ODM_INFO("Good feature matches: %s" % len(matches))

    # Extract location of good matches
    points_image = np.zeros((len(matches), 2), dtype=np.float32)
    points_align_image = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_image[i, :] = kp_image[match.queryIdx].pt
        points_align_image[i, :] = kp_align_image[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points_image, points_align_image, cv2.RANSAC)
    return h

def find_rig_homography(photo, align_photo, image_gray, align_image_gray):
    image_undistorted_gray = photo.undistorted(image_gray)
    align_image_undistorted_gray = align_photo.undistorted(align_image_gray)

    # compute homography matrices
    M_ig = find_features_homography(image_gray, image_undistorted_gray)
    M_aig = find_features_homography(align_image_gray, align_image_undistorted_gray)
    M = photo.get_homography(align_photo)

    if M_ig is None:
        M_ig = np.eye(3, 3, dtype=np.float32)
        log.ODM_INFO("Cannot find feature homography between the raw image and undistorted image for %s, use identity matrix instead" % photo.filename)

    if M_aig is None:
        M_aig = np.eye(3, 3, dtype=np.float32)
        log.ODM_INFO("Cannot find feature homography between the raw image and undistorted image for %s, use identity matrix instead" % align_photo.filename)

    # log.ODM_INFO("%s --> %s transform matrices: M_src=%s, M_dst=%s, M_src_dst=%s" % (photo.filename, align_photo.filename, M_ig, M_aig, M))

    warp_matrix = np.array(np.dot(np.linalg.inv(M_aig), np.dot(M, M_ig)))
    warp_matrix /= warp_matrix[2,2]
    return warp_matrix

def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    if min is not None and max is not None:
        norm = (im - min) / (max-min)
    else:
        cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm[norm<0.0] = 0.0
    norm[norm>1.0] = 1.0
    return norm

def gradient(im, ksize=5):
    im = local_normalize(im)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=ksize)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=ksize)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def local_normalize(im):
    norm = img_as_ubyte(normalize(im))
    width, _ = im.shape
    disksize = int(width/5)
    if disksize % 2 == 0:
        disksize = disksize + 1
    selem = disk(disksize)
    im = rank.equalize(norm, selem=selem)
    return im

def align_image(image, warp_matrix, dimension, flags=cv2.INTER_LINEAR):
    if warp_matrix.shape == (3, 3):
        return cv2.warpPerspective(image, warp_matrix, dimension, flags=flags)
    else:
        return cv2.warpAffine(image, warp_matrix, dimension, flags=flags)


def to_8bit(image, force_normalize=False):
    if not force_normalize and image.dtype == np.uint8:
        return image

    # Convert to 8bit
    try:
        data_range = np.iinfo(image.dtype)
        min_value = 0
        value_range = float(data_range.max) - float(data_range.min)
    except ValueError:
        # For floats use the actual range of the image values
        min_value = float(image.min())
        value_range = float(image.max()) - min_value

    image = image.astype(np.float32)
    image -= min_value
    image *= 255.0 / value_range
    np.around(image, out=image)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)

    return image

# Need further review the use of this method
def resize_match(image, dimension):
    h, w = image.shape[0], image.shape[1]
    mw, mh = dimension

    if w != mw or h != mh:
        fx = mw/w
        fy = mh/h
        image = cv2.resize(image, None,
                fx=fx,
                fy=fx,
                interpolation=(cv2.INTER_AREA if (fx < 1.0 and fy < 1.0) else cv2.INTER_LANCZOS4))

    return image

######################################################################################################################
# Custom image band vignetting handler
######################################################################################################################
def compute_band_vignette_map(multi_camera):
    band_vignette_map = {}

    for band in multi_camera:
        photos = get_photos_by_band(multi_camera, band['name'])
        ref_photo = photos[0]

        if ref_photo.camera_make == "Parrot" and ref_photo.camera_model == "Sequoia":
            log.ODM_INFO("Computing %s band vignetting coefficients" % band['name'])
            vignetting_coefs = ref_photo.get_vignetting_coefficients_sequoia()
            if vignetting_coefs is not None:
                band_vignette_map[band['name']] = vignetting_coefs
                log.ODM_INFO("%s band's vignetting coefficients: %s" % (band['name'], band_vignette_map.get(band['name'])))

        band_vignette_map[band['name']] = None

    return band_vignette_map
