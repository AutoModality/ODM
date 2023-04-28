import os, math

from opendm import log
from opendm import io
from opendm import system
from opendm import context
from opendm import mesh
from opendm import gsd
from opendm import types
from opendm.dem import commands
from opendm.dem import pdal

class ODMeshingStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        # define paths and create working directories
        system.mkdir_p(tree.odm_meshing)

        # Create full 3D model unless --skip-3dmodel is set
        if not args.skip_3dmodel:
            if not io.file_exists(tree.odm_mesh) or self.rerun():
                log.ODM_INFO('Writing ODM Mesh file in: %s' % tree.odm_mesh)

                mesh.screened_poisson_reconstruction(tree.filtered_point_cloud,
                    tree.odm_mesh,
                    depth=self.params.get('oct_tree'),
                    samples=self.params.get('samples'),
                    maxVertexCount=self.params.get('max_vertex'),
                    pointWeight=self.params.get('point_weight'),
                    threads=max(1, self.params.get('max_concurrency') - 1)), # poissonrecon can get stuck on some machines if --threads == all cores
            else:
                log.ODM_WARNING('Found a valid ODM Mesh file in: %s' %
                                tree.odm_mesh)
        
        self.update_progress(50)

        # Always generate a 2.5D mesh for texturing and orthophoto projection
        # unless --use-3dmesh is set.
        if not args.use_3dmesh:
            if not io.file_exists(tree.odm_25dmesh) or self.rerun():

                log.ODM_INFO('Writing ODM 2.5D Mesh file in: %s' % tree.odm_25dmesh)

                pc_quality_scale = {
                    'ultra': 2.0, # capped to 2X
                    'high': 4.0,
                    'medium': 8.0,
                    'low': 16.0,
                    'lowest': 16.0 # capped to 16X
                }
                if args.texturing_use_dtm:
                    pc_quality_scale = {
                        'ultra': 1.0,
                        'high': 2.0,
                        'medium': 4.0,
                        'low': 8.0,
                        'lowest': 16.0
                    }
                dem_resolution = gsd.cap_resolution(args.dem_resolution, tree.opensfm_reconstruction,
                                                    gsd_scaling=pc_quality_scale[args.pc_quality],
                                                    ignore_gsd=args.ignore_gsd,
                                                    ignore_resolution=(not reconstruction.is_georeferenced()) and args.ignore_gsd,
                                                    has_gcp=reconstruction.has_gcp()) / 100.0
                if args.fast_orthophoto:
                    dem_resolution *= 2.0

                radius_steps = [str(dem_resolution * math.sqrt(2)), str(dem_resolution * 2)]

                log.ODM_INFO('ODM 2.5D DEM resolution: %s' % dem_resolution)

                dem_input = tree.filtered_point_cloud
                if args.texturing_use_dtm:
                    pdal.run_pdaltranslate_smrf(tree.filtered_point_cloud,
                                                tree.filtered_point_cloud_classified,
                                                args.smrf_scalar,
                                                args.smrf_slope,
                                                args.smrf_threshold,
                                                args.smrf_window)
                    dem_input = tree.filtered_point_cloud_classified

                mesh.create_25dmesh(dem_input, tree.odm_25dmesh,
                        radius_steps=radius_steps,
                        dsm_resolution=dem_resolution, 
                        depth=self.params.get('oct_tree'),
                        maxVertexCount=self.params.get('max_vertex'),
                        samples=self.params.get('samples'),
                        available_cores=args.max_concurrency,
                        method='poisson' if args.fast_orthophoto else 'gridded',
                        smooth_dsm=True,
                        use_dtm=args.texturing_use_dtm)
                
                if io.file_exists(tree.filtered_point_cloud_classified):
                    os.remove(tree.filtered_point_cloud_classified)
            else:
                log.ODM_WARNING('Found a valid ODM 2.5D Mesh file in: %s' %
                                tree.odm_25dmesh)

