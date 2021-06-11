import os
import matplotlib as mpl
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
import cv2
import imageio
import roosts.utils.file_util as fileUtil
from tqdm import tqdm

class Visualizer:

    """
        Visualize the detection and tracking results
    """


    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height


    def draw_detections(self,
                        image_paths, 
                        detections, 
                        outdir, 
                        score_thresh=0.005, 
                        save_gif=True,
                        vis_track=False,
                        vis_track_after_NMS=True):
        """ 
            Draws detections on the images
        
            Args:
                image_paths: absolute path of images, type: list
                detections:  the output of detector or tracker with the structure 
                             {"scanname":xx, "im_bbox": xx, "det_ID": xx, 'det_score': xx}
                             type: list of dict
                outdir: path to save images
                score_thresh: only display bbox with score higher than threshold
                save_image: store image 
                save_gif:   save image sequence as gif on a single station in daily basis

            Returns: 
                image with bboxes
        """
        fileUtil.mkdir(outdir)
        outpaths = []

        if not vis_track:
            # if visualize track, some detections are predicted by Kalman filter which may not have det score
            detections = [det for det in detections if det["det_score"] >= score_thresh]
    
        if vis_track and vis_track_after_NMS:
             # the track is not suppressed by NMS
            detections = [det for det in detections if ("track_NMS" in det.keys()) and (not det["track_NMS"])]

        for image_path in tqdm(image_paths, desc="Visualizing"):

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            scanname = os.path.splitext(os.path.basename(image_path))[0]
            dets = [det for det in detections if det["scanname"] in scanname]
            outname = os.path.join(outdir, os.path.basename(image_path))
            self.overlay_detections(image, dets, outname)
            outpaths.append(outname)

        if save_gif:
            gif_path = os.path.join(outdir, scanname.split("_")[0] + '.gif')
            self.save_gif(outpaths, gif_path)
            return gif_path
            
        return True

    def overlay_detections(self, image, detections, outname):
        """ Overlay bounding boxes on images  """

        fig = mplfigure.Figure(frameon=False)
        dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width + 1e-2 ) / dpi,
            (self.height + 1e-2 ) / dpi,
        )
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.imshow(image, extent=(0, self.width, self.height, 0), interpolation="nearest")
       
        for det in detections:
            x, y, r = det["im_bbox"]
            score = det["det_score"]

            ax.add_patch(
                plt.Rectangle((x-r, y-r), 
                               2*r,
                               2*r, fill=False,
                               edgecolor= '#FF00FF', linewidth=3)
                )
            if "track_ID" in det.keys():
                ax.text(x-r, y-r-2,
                        '{:d}'.format(det["track_ID"]),
                        bbox=dict(facecolor='blue', alpha=0.7),
                        fontsize=14, color='white')
            else:
                ax.text(x-r, y-r-2,
                        '{:.3f}'.format(score),
                        bbox=dict(facecolor='#FF00FF', alpha=0.7),
                        fontsize=14, color='white')
        fig.savefig(outname) 
        plt.close()

    

    def save_gif(self, image_paths, outpath):
        """ 
            imageio may load the image in a different format from matplotlib,
            so I just reload the images from local disk by imageio.imread 
        """
        seq = []
        image_paths.sort()
        for image_path in image_paths:
            seq.append(imageio.imread(image_path))
        kargs = {"duration": 0.5}
        imageio.mimsave(outpath, seq, "GIF", **kargs)
            

    def generate_web_files(self, detections, tracks, outpath):
        
        det_dict = {}
        for det in detections:
            det_dict[det["det_ID"]] = det
        
        with open(outpath, 'w+') as f:
            f.write('track_id,filename,from_sunrise,det_score,x,y,r,lon,lat,radius,is_rain\n')
            for track in tqdm(tracks, desc="Write tracks into csv"):
                # remove the tail of tracks (which are generated from Kalman filter instead of detector)
                for idx in range(len(track["det_or_pred"])-1, -1, -1):
                    if track["det_or_pred"][idx]:
                        last_pred_idx = idx
                        break
                # do not report the tail of tracks
                for idx, det_ID in enumerate(track["det_IDs"]):
                    if idx > last_pred_idx:
                        break
                    det = det_dict[det_ID]
                    if (("windfarm" in det.keys()) and det["windfarm"]):
                        continue
                    f.write('{:d},{:s},{:d},{:.3f},{:.2f},{:2f},{:2f},{:.2f},{:2f},{:2f},{:d}\n'.format(
                        det["track_ID"], det["scanname"], int(det["from_sunrise"]), 
                        det["det_score"], det["im_bbox"][0], det["im_bbox"][1], det["im_bbox"][2], 
                        det["geo_bbox"][0], det["geo_bbox"][1], det["geo_bbox"][2],
                        det["rain"]))

