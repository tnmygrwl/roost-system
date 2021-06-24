import os
import torch
print("torch.get_num_threads: ", torch.get_num_threads())
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
import logging
import time
import argparse

here = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, required=True, help="a single station name, eg. KDOX")
parser.add_argument('--start', type=str, required=True, help="the first day to process, eg. 20101001")
parser.add_argument('--end', type=str, required=True, help="the last day to process, eg. 20101001")
parser.add_argument('--ckpt_path', type=str,
                    default=f"{here}/../checkpoints/entire_c4_9anchor.pth")
parser.add_argument('--data_root', type=str,
                    default=f"{here}/../roosts_data")
args = parser.parse_args()

######################## define station and date ############################
request = {"station": args.station, "date": (args.start, args.end)}


######################## paths and directories ############################
scan_dir            = os.path.join(args.data_root, 'scans')  # save raw scans downloaded from AWS
npz_dir             = os.path.join(args.data_root, 'arrays') # npz files and rendered reflective/radio velocity
log_root_dir        = os.path.join(args.data_root, 'logs')
vis_det_dir         = os.path.join(args.data_root, 'vis_dets') # visualization of detections from detection model
vis_NMS_MERGE_track_dir   = os.path.join(args.data_root, 'vis_NMS_MERGE_tracks') # vis of tracks after NMS & merge
ui_dir              = os.path.join(args.data_root, 'ui') # save files for website ui visualization
ui_img_dir          = os.path.join(ui_dir, 'img')
scan_and_track_dir  = os.path.join(ui_dir, "scans_and_tracks")
os.makedirs(scan_and_track_dir, exist_ok=True)


######################## Initialize models ############################
downloader = Downloader(min_before_sunrise=30, min_after_sunrise=90, log_dir=log_root_dir)
downloader.set_request(request, scan_dir)
renderer = Renderer(npz_dir, ui_img_dir)
detector = Detector(
    args.ckpt_path,
    anchor_sizes = [[16, 32, 48, 64, 80, 96, 112, 128, 144]], # [[32], [64], [128], [256], [512]] for FPN
    use_gpu = torch.cuda.is_available()
)
tracker = Tracker()
visualizer = Visualizer()
postprocess = Postprocess(
    imsize = 600,
    geosize = 300000,
    clean_windfarm = True,
    clean_rain = True
)
n_existing_tracks = 0


######################## process radar data ############################
print("Total number of days: %d" % len(downloader))
print(f"---------------------- Day 1 -----------------------\n")

######################## (1) Download data ############################
for day_idx, downloader_outputs in enumerate(downloader):

    if downloader_outputs is StopIteration:
        break
    else:
        scan_paths, start_time, key_prefix, logger = downloader_outputs
        year, month, _, _ = key_prefix.split("/")

    ######################## (2) Render data ############################
    """
        npz_files: for detection module to load/preprocess data
        img_files: for visualization
        scan_names: for tracking module to know the full image set
    """

    npz_files, img_files, scan_names = renderer.render(scan_paths, key_prefix, logger)
    # fileUtil.delete_files(scan_paths)

    with open(os.path.join(
            scan_and_track_dir, f'scans_{args.station}_{args.start}_{args.end}.txt'
    ), "a+") as f:
        f.writelines([scan_name + "\n" for scan_name in scan_names])

    if len(npz_files) == 0:
        print()
        if day_idx + 2 <= len(downloader):
            print(f"---------------------- Day {day_idx + 2} -----------------------\n")
        continue

    ######################## (3) Run detection models on the data ############################
    detections = detector.run(npz_files)
    logger.info(f'[Detection Done] {len(detections)} detections')

    ######################## (4) Run tracking on the detections  ############################
    """
        in some cases, the detector does not find any roosts,
        therefore, we need "scan_names" (a name list of all scans) to let the tracker find some using tracking info
        NMS over tracks is applied to remove duplicated tracks, not sure if it's useful with new detection model
    """
    tracked_detections, tracks = tracker.tracking(scan_names, copy.deepcopy(detections))
    logger.info(f'[Tracking Done] {len(tracks)} tracks with {len(tracked_detections)} tracked detections')

    # ######################## (5) Postprocessing  ############################
    # """
    #     (1) convert image coordinates to geometric coordinates;
    #     (2) clean up the false positives due to windfarm and rain using auxiliary information
    # """
    cleaned_detections, tracks = postprocess.annotate_detections(copy.deepcopy(tracked_detections),
                                                          copy.deepcopy(tracks),
                                                          npz_files)
    logger.info(f'[Postprocessing Done] {len(cleaned_detections)} cleaned detections')

    ######################## (6) Visualize the detection and tracking results  ############################
    
    """ visualize detections under multiple thresholds of detection score"""
    gif_path1 = visualizer.draw_dets_multi_thresh(
        img_files, copy.deepcopy(detections), os.path.join(vis_det_dir, args.station, year, month)
    )

    """ visualize results after NMS and merging on tracks"""
    gif_path2 = visualizer.draw_tracks_multi_thresh(
        img_files, copy.deepcopy(tracked_detections), copy.deepcopy(tracks),
        os.path.join(vis_NMS_MERGE_track_dir, args.station, year, month)
    )
    
    # generate a website file
    station_day = scan_names[0][:12]
    n_existing_tracks = visualizer.generate_web_files(
        cleaned_detections, tracks, os.path.join(
            scan_and_track_dir, f'tracks_{args.station}_{args.start}_{args.end}.txt'
        ), n_existing_tracks=n_existing_tracks
    )

    end_time = time.time()
    logger.info(f'[Finished] running the system on {station_day}; '
                f'total time elapse: {end_time - start_time}')

    print("Total time elapse: {}".format(end_time - start_time))
    print()
    if day_idx + 2 <= len(downloader):
        print(f"-------------------- Day {day_idx + 2} --------------------\n")

