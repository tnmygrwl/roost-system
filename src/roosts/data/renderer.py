import os
import numpy as np
from wsrlib import pyart, radar2mat
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import cv2
import roosts.utils.file_util as fileUtil
import glob
from tqdm import tqdm

ARRAY_VERSION       = "v0.1.0" # corresponding to arrays defined by the following lines
ARRAY_Y_DIRECTION   = "xy" # default radar direction, y is first dim (row), large y is north, row 0 is south
ARRAY_R_MAX         = 150000.0
ARRAY_DIM           = 600
ARRAY_ATTRIBUTES    = ["reflectivity", "velocity", "spectrum_width"]
ARRAY_ELEVATIONS    = [0.5, 1.5, 2.5, 3.5, 4.5]
ARRAY_RENDER_CONFIG = {"ydirection":          ARRAY_Y_DIRECTION,
                       "fields":              ARRAY_ATTRIBUTES,
                       "coords":              "cartesian",
                       "r_min":               2125.0,       # default: first range bin of WSR-88D
                       "r_max":               ARRAY_R_MAX,  # 459875.0 default: last range bin
                       "r_res":               250,          # default: super-res gate spacing
                       "az_res":              0.5,          # default: super-res azimuth resolution
                       "dim":                 ARRAY_DIM,    # num pixels on a side in Cartesian rendering
                       "sweeps":              None,
                       "elevs":               ARRAY_ELEVATIONS,
                       "use_ground_range":    True,
                       "interp_method":       'nearest'}
DUALPOL_DIM             = 600
DUALPOL_ATTRIBUTES      = ["differential_reflectivity", "cross_correlation_ratio", "differential_phase"]
DUALPOL_ELEVATIONS      = [0.5, 1.5, 2.5, 3.5, 4.5]
DUALPOL_RENDER_CONFIG   = {"ydirection":          ARRAY_Y_DIRECTION,
                           "fields":              DUALPOL_ATTRIBUTES,
                           "coords":              "cartesian",
                           "r_min":               2125.0,       # default: first range bin of WSR-88D
                           "r_max":               ARRAY_R_MAX,  # default 459875.0: last range bin
                           "r_res":               250,          # default: super-res gate spacing
                           "az_res":              0.5,          # default: super-res azimuth resolution
                           "dim":                 DUALPOL_DIM,  # num pixels on a side in Cartesian rendering
                           "sweeps":              None,
                           "elevs":               DUALPOL_ELEVATIONS,
                           "use_ground_range":    True,
                           "interp_method":       "nearest"}

####### render as images #######
CHANNELS = [("reflectivity", 0.5), ("velocity", 0.5)]
# visualization settings
COLOR_ARRAY = [
    '#006400', # for not scaled boxes
    '#FF00FF', # scaled to RCNN then to user factor 0.7429 which is sheldon average
    '#800080',
    '#FFA500',
    '#FFFF00'
]
NORMALIZERS = {
        'reflectivity':              pltc.Normalize(vmin=  -5, vmax= 35),
        'velocity':                  pltc.Normalize(vmin= -15, vmax= 15),
        'spectrum_width':            pltc.Normalize(vmin=   0, vmax= 10),
        'differential_reflectivity': pltc.Normalize(vmin=  -4, vmax= 8),
        'differential_phase':        pltc.Normalize(vmin=   0, vmax= 250),
        'cross_correlation_ratio':   pltc.Normalize(vmin=   0, vmax= 1.1)
}

class Renderer:

    """ 
        Extract radar products from raw radar scans, 
        save the products in npz files, 
        save the dualpol data for postprocessing,
        render ref1 and rv1 for visualization     
        delete the radar scans 

        input: paths of scan list, outpath npz files and 
        output: paths of npz file used for detector

    """

    def __init__(self, 
                 outpath=None,
                 array_render_config=ARRAY_RENDER_CONFIG, 
                 dualpol_render_config=DUALPOL_RENDER_CONFIG):

        self.outpath = outpath
        self.npzpath = os.path.join(outpath, 'npz')
        self.imgpath = os.path.join(outpath, 'img')
        self.array_render_config = array_render_config 
        self.dualpol_render_config = dualpol_render_config

        fileUtil.mkdir(self.outpath)
        fileUtil.mkdir(self.npzpath)
        fileUtil.mkdir(self.imgpath)


    def render(self, scan_paths):

        npz_files = []
        img_files = []
        scan_names = []

        logger = logging.getLogger(__name__)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s [ %(name)s ] : %(message)s')
            formatter.converter = time.gmtime
            logger.setLevel(logging.INFO)

        for scan_file in tqdm(scan_paths, desc="Rendering"):
            
            scan = os.path.splitext(os.path.basename(scan_file))[0]

            npz_path = os.path.join(self.npzpath, f'{scan}.npz')
            ref1_path = os.path.join(self.imgpath, f'{scan}_ref1.jpg')
            rv1_path = os.path.join(self.imgpath, f'{scan}_rv1.jpg')

            if os.path.exists(npz_path) and os.path.exists(ref1_path):
                npz_files.append(npz_path)
                img_files.append(ref1_path)
                scan_names.append(scan)
                continue

            arrays = {}

            try:
                radar = pyart.io.read_nexrad_archive(scan_file)
                logger.info('Loaded scan %s' % scan)
            except Exception as ex:
                # logger.error('Exception while loading scan %s - %s' % (scan, str(ex)))
                continue

            try:
                data, _, _, y, x = radar2mat(radar, **self.array_render_config)
                logger.info('Rendered a npy array from scan %s' % scan)
                if data.shape != (len(self.array_render_config["fields"]), len(self.array_render_config["elevs"]),
                                  self.array_render_config["dim"], self.array_render_config["dim"]):
                    logger.info(f"  Unexpectedly, its shape is {data.shape}.")
                arrays["array"] = data
            except Exception as ex:
                logger.error('Exception while rendering a npy array from scan %s - %s' % (scan, str(ex)))

            try:
                data, _, _, y, x = radar2mat(radar, **self.dualpol_render_config)
                logger.info('Rendered a dualpol npy array from scan %s' % scan)
                if data.shape != (len(self.dualpol_render_config["fields"]), len(self.dualpol_render_config["elevs"]),
                                  self.dualpol_render_config["dim"], self.dualpol_render_config["dim"]):
                    logger.info(f"  Unexpectedly, its shape is {data.shape}.")
                arrays["dualpol_array"] = data
            except Exception as ex:
                # logger.error('Exception while rendering a dualpol npy array from scan %s - %s' % (scan, str(ex)))
                pass

            if len(arrays) > 0:
                np.savez_compressed(npz_path, **arrays)
                self.render_img(arrays["array"], scan, ref1_path, rv1_path) # render ref1 and rv1 images 
                npz_files.append(npz_path)
                img_files.append(ref1_path)
                scan_names.append(scan)

        return npz_files, img_files, scan_names 


    def render_img(self, array, scan, ref1_path, rv1_path):
        # input: numpy array containing radar products
        outpath = {"reflectivity": ref1_path, "velocity": rv1_path}
        attributes = self.array_render_config['fields']
        elevations = self.array_render_config['elevs']
        for i, (attr, elev) in enumerate(CHANNELS):
            cm = plt.get_cmap(pyart.config.get_field_colormap(attr))
            rgb = cm(NORMALIZERS[attr](array[attributes.index(attr), elevations.index(elev), :, :]))
            rgb = (rgb * 255).astype(np.uint8) # cv2 requires uint8 to write an image
            rgb = rgb[:, :, :3]  # omit the fourth alpha dimension, NAN are black but not white
            cv2.imwrite(outpath[attr], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    
    scan_root = './scans/2021/05/24/KDOX/'
    outpath = './arrays'
    scan_paths = glob.glob(scan_root + '*')
    renderer = Renderer(outpath)
    renderer.render(scan_paths)


