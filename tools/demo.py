import os
import torch
print(f"torch.get_num_threads: {torch.get_num_threads()}", flush=True)
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
parser.add_argument('--sun_activity', type=str, default="sunrise", help="time window around sunrise or sunset")
parser.add_argument('--min_before', type=int, default=30,
                    help="process scans at most these minutes before the selected sun activity")
parser.add_argument('--min_after', type=int, default=90,
                    help="process scans at most these minutes after the selected sun activity")
parser.add_argument('--ckpt_path', type=str, help="detection model checkpoint path",
                    default=f"{here}/../checkpoints/entire_c4_9anchor.pth")
parser.add_argument('--data_root', type=str, help="directory for all outputs",
                    default=f"{here}/../roosts_data")
parser.add_argument('--gif_vis', action='store_true', help="generate gif visualization")
parser.add_argument('--no_ui', action='store_true', help="do not generate files for UI")
args = parser.parse_args()
print(args, flush=True)

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
downloader = Downloader(
    sun_activity=args.sun_activity, min_before=args.min_before, min_after=args.min_after,
    log_dir=log_root_dir,
)
downloader.set_request(request, scan_dir)
renderer = Renderer(npz_dir, ui_img_dir)
detector = Detector(
    args.ckpt_path,
    imsize = 1200,
    anchor_sizes = [[16, 32, 48, 64, 80, 96, 112, 128, 144]], # [[32], [64], [128], [256], [512]] for FPN
    nms_thresh = 0.3,
    score_thresh = 0.1,
    config_file = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    use_gpu = torch.cuda.is_available(),
)
tracker = Tracker()
visualizer = Visualizer(sun_activity=args.sun_activity)
postprocess = Postprocess(
    imsize = 600,
    geosize = 300000,
    sun_activity=args.sun_activity,
    clean_windfarm = True,
    clean_rain = True
)
n_existing_tracks = 0


######################## process radar data ############################
print("Total number of days: %d" % len(downloader), flush=True)

######################## (1) Download data ############################
for day_idx, downloader_outputs in enumerate(downloader):

    if downloader_outputs is StopIteration:
        break
    else:
        date_string, date_station_prefix, logger, scan_paths, start_time = downloader_outputs
        year, month, _, _ = date_station_prefix.split("/")
        print(f"-------------------- Day {day_idx+1}: {date_string} --------------------\n", flush=True)

    ######################## (2) Render data ############################
    """
        npz_files: for detection module to load/preprocess data
        img_files: for visualization
        scan_names: for tracking module to know the full image set
    """

    npz_files, img_files, scan_names = renderer.render(scan_paths, date_station_prefix, logger)
    fileUtil.delete_files(scan_paths)

    with open(os.path.join(
            scan_and_track_dir, f'scans_{args.station}_{args.start}_{args.end}.txt'
    ), "a+") as f:
        f.writelines([scan_name + "\n" for scan_name in scan_names])

    if len(npz_files) == 0:
        end_time = time.time()
        logger.info(f'[Passed] no successfully rendered scan for {args.station} {date_string}; '
                    f'total time elapse: {end_time - start_time}')
        print(f"No successfully rendered scan.\nTotal time elapse: {end_time - start_time}\n", flush=True)
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

    # generate gif visualization
    if args.gif_vis:
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
    if not args.no_ui:
        n_existing_tracks = visualizer.generate_web_files(
            cleaned_detections, tracks, os.path.join(
                scan_and_track_dir, f'tracks_{args.station}_{args.start}_{args.end}.txt'
            ), n_existing_tracks=n_existing_tracks
        )

    end_time = time.time()
    logger.info(f'[Finished] running the system for {args.station} {date_string}; '
                f'total time elapse: {end_time - start_time}')
    print(f"Total time elapse: {end_time - start_time}\n", flush=True)

