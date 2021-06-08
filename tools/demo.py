import os
from roosts.data.downloader import Downloader
from roosts.data.renderer import Renderer
from roosts.detection.detector import Detector
from roosts.tracking.tracker import Tracker
from roosts.utils.visualizer import Visualizer
from roosts.utils.postprocess import Postprocess
import roosts.utils.file_util as fileUtil
import copy
import warnings
warnings.filterwarnings("ignore")
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, required=True, help="a single station name, eg. KDOX")
parser.add_argument('--start', type=str, required=True, help="the first day to process, eg. 20101001")
parser.add_argument('--end', type=str, required=True, help="the last day to process, eg. 20101001")
args = parser.parse_args()

######################## define station and date ############################
request = {"station": args.station, "date": (args.start, args.end)}


######################## paths and directories ############################
ckpt_path           = "../checkpoints/entire_lr_0.001.pth"
windfarm_database   = "../src/roosts/utils/uswtdb_v1_3_20190107.csv"
data_root           = "../roosts_data"
scan_dir            = os.path.join(data_root, 'scans')  # save raw scans downloaded from AWS
npz_dir             = os.path.join(data_root, 'arrays') # npz files and rendered reflective/radio velocity
vis_det_dir         = os.path.join(data_root, 'vis_dets') # visualization of detections from detection model
vis_cleaned_det_dir = os.path.join(data_root, 'vis_cleaned_dets') # visualization of detections after removing rain / windfarm
vis_track_dir       = os.path.join(data_root, 'vis_tracks') # visualization of tracks
vis_NMS_track_dir   = os.path.join(data_root, 'vis_NMS_tracks') # visualization of tracks after NMS
roosts_ui_data_dir  = os.path.join(data_root, 'roosts_ui_data') # save files for website ui visualization


######################## initialize models ############################
downloader  = Downloader(min_before_sunrise=30, min_after_sunrise=90)
downloader.set_request(request, scan_dir)
renderer    = Renderer(npz_dir, roosts_ui_data_dir)
detector    = Detector(ckpt_path, use_gpu=True)
tracker     = Tracker()
visualizer  = Visualizer()
postprocess = Postprocess(imsize=600,
                           geosize=300000,
                           windfarm_database=windfarm_database,
                           clean_windfarm=True,
                           clean_rain=True)


######################## process radar data ############################
print("Total number of days: %d" % len(downloader))
print(f"---------------------- Day 1 -----------------------\n")
for day_idx, scan_paths in enumerate(downloader):
    start_time = time.time()

    ######################## (1) Download data ############################
    if type(scan_paths) not in (list,): break


    ######################## (2) Render data ############################
    """
        npz_files: for detection module to load/preprocess data
        img_files: for visualization
        scan_names: for tracking module to know the full image set
    """

    npz_files, img_files, scan_names = renderer.render(scan_paths)
    # fileUtil.delete_files(scan_paths)

    if len(npz_files) == 0:
        print()
        print(f"---------------------- Day {day_idx+2} -----------------------\n")
        continue

    ######################## (3) Run detection models on the data ############################
    detections = detector.run(npz_files)

    ######################## (4) Run tracking on the detections  ############################
    """
        in some cases, the detector does not find any roosts,
        therefore, we need "scan_names" (a name list of all scans) to let the tracker find some using tracking info
        NMS over tracks is applied to remove duplicated tracks, not sure if it's useful with new detection model
    """
    tracked_detections, tracks = tracker.tracking(scan_names, copy.deepcopy(detections))
    #
    # ######################## (5) Postprocessing  ############################
    # """
    #     (1) convert image coordinates to geometric coordinates;
    #     (2) clean up the false positives due to windfarm and rain using auxiliary information
    # """
    cleaned_detections = postprocess.annotate_detections(copy.deepcopy(tracked_detections),
                                                          copy.deepcopy(tracks),
                                                          npz_files)
    #
    # ######################## (6) Visualize the detection and tracking results  ############################
    #
    # import pdb; pdb.set_trace()

    gif_path1 = visualizer.draw_detections(img_files, copy.deepcopy(detections),
                                vis_det_dir, score_thresh=0.000, save_gif=True)
    # gif_path2 = visualizer.draw_detections(img_files, copy.deepcopy(tracked_detections),
    #                            vis_track_dir,  save_gif=True,  vis_track=True)
    #
    # gif_path3 = visualizer.draw_detections(img_files, copy.deepcopy(cleaned_detections),
    #                             vis_cleaned_det_dir, score_thresh=0.000, save_gif=True)
    #
    gif_path4 = visualizer.draw_detections(img_files, copy.deepcopy(cleaned_detections),
                                 vis_NMS_track_dir, save_gif=True, vis_track=True, vis_track_after_NMS=True)
    #
    # # generate a website file
    station_day = scan_names[0][:12]
    # visualizer.generate_web_files(detections, tracks, os.path.join(roosts_ui_data_dir, f'{station_day}.txt'))
    visualizer.generate_web_files(cleaned_detections, tracks, os.path.join(roosts_ui_data_dir, f'{station_day}.txt'))

    #
    #
    # print("Total time elapse: {}".format(time.time() - start_time))
    #
    # # import pdb; pdb.set_trace()
    # print()
    # print(f"-------------------- Day {day_idx+2} --------------------\n")

