import os
import rasterio

from datetime import datetime
from osgeo import gdal
from opendm import io
from opendm import log
from opendm import types
from opendm import photo
from opendm.utils import copy_paths, get_processing_results_paths
from opendm.ogctiles import build_3dtiles

class ODMPostProcess(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        log.ODM_INFO("Post Processing")

        rasters = [tree.odm_orthophoto_tif,
                    tree.path("odm_dem", "dsm.tif"),
                    tree.path("odm_dem", "dtm.tif")]

        mean_capture_time = photo.find_mean_utc_time(reconstruction.photos)
        mean_capture_dt = None
        if mean_capture_time is not None:
            mean_capture_dt = datetime.fromtimestamp(mean_capture_time).strftime('%Y:%m:%d %H:%M:%S') + '+00:00'

        # Add TIFF tags
        for product in rasters:
            if os.path.isfile(product):
                log.ODM_INFO("Adding TIFFTAGs to {}".format(product))
                with rasterio.open(product, 'r+') as rst:
                    if mean_capture_dt is not None:
                        rst.update_tags(TIFFTAG_DATETIME=mean_capture_dt)
                    rst.update_tags(TIFFTAG_SOFTWARE='ODM {}'.format(log.odm_version()))

        # GCP info
        if not outputs['large']:
            # TODO: support for split-merge?

            # Embed GCP info in 2D results via
            # XML metadata fields
            gcp_gml_export_file = tree.path("odm_georeferencing", "ground_control_points.gml")

            if reconstruction.has_gcp() and io.file_exists(gcp_gml_export_file):
                skip_embed_gcp = False
                gcp_xml = ""

                with open(gcp_gml_export_file) as f:
                    gcp_xml = f.read()

                for product in rasters:
                    if os.path.isfile(product):
                        ds = gdal.Open(product)
                        if ds is not None:
                            if ds.GetMetadata('xml:GROUND_CONTROL_POINTS') is None or self.rerun():
                                ds.SetMetadata(gcp_xml, 'xml:GROUND_CONTROL_POINTS')
                                ds = None
                                log.ODM_INFO("Wrote xml:GROUND_CONTROL_POINTS metadata to %s" % product)
                            else:
                                skip_embed_gcp = True
                                log.ODM_WARNING("Already embedded ground control point information")
                                break
                        else:
                            log.ODM_WARNING("Cannot open %s for writing, skipping GCP embedding" % product)

        # Generate normalized DSM if both DSM and DTM exist
        dsm = tree.path("odm_dem", "dsm.tif")
        dtm = tree.path("odm_dem", "dtm.tif")
        ndsm = tree.path("odm_dem", "ndsm.tif")
        if os.path.isfile(dsm) and os.path.isfile(dtm):
            try:
                log.ODM_INFO("Generating normalized DSM: %s" % ndsm)
                dsm_ds = gdal.Open(dsm)
                dsm_band = dsm_ds.GetRasterBand(1)
                dsm_array = np.ma.masked_equal(dsm_band.ReadAsArray(), dsm_band.GetNoDataValue())

                dtm_ds = gdal.Open(dtm)
                dtm_band = dtm_ds.GetRasterBand(1)
                dtm_array = np.ma.masked_equal(dtm_band.ReadAsArray(), dtm_band.GetNoDataValue())

                # nDSM = DSM - DTM
                ndsm_data = dsm_array - dtm_array
                ndsm_ds = gdal_array.SaveArray(ndsm_data, ndsm, "GTIFF", dsm_ds)

                # set same nodata value as DSM
                no_data = dsm_band.GetNoDataValue()
                ndsm_ds.GetRasterBand(1).SetNoDataValue(no_data)

                # close the tiff files
                dsm_ds = None
                dtm_ds = None
                ndsm_ds = None

                if os.path.isfile(ndsm):
                    log.ODM_INFO("Generating normalized DSM finished.")
                    convert_to_cogeo(ndsm)
                else:
                    log.ODM_WARNING("Generating normalized DSM failed.")

            except Exception as e:
                log.ODM_WARNING("Cannot generate normalized DSM. %s" % str(e))


        if getattr(args, '3d_tiles'):
            build_3dtiles(args, tree, reconstruction, self.rerun())

        if args.copy_to:
            try:
                copy_paths([os.path.join(args.project_path, p) for p in get_processing_results_paths()], args.copy_to, self.rerun())
            except Exception as e:
                log.ODM_WARNING("Cannot copy to %s: %s" % (args.copy_to, str(e)))
