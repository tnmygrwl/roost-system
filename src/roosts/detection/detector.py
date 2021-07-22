import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import matplotlib.colors as pltc
import numpy as np
import os
from tqdm import tqdm

class Detector:

    def __init__(self, 
                 ckpt_path,             # path of pretrained detector
                 imsize = 1200,         # input image size
                 anchor_sizes = [[16, 32, 48, 64, 80, 96, 112, 128, 144]],  # predefined anchor sizes
                 nms_thresh = 0.3,      # non-maximum suppression
                 score_thresh = 0.1,    # filter out detections with score lower than score_thresh
                 config_file = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml", # define the detection model
                 use_gpu = False,       # GPU or CPU
    ):

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.INPUT.MIN_SIZE_TEST = imsize
        cfg.INPUT.MAX_SIZE_TEST = imsize
        cfg.INPUT.FORMAT = 'BGR'  # just so scan channels are not altered
        cfg.MODEL.WEIGHTS = ckpt_path
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_sizes
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
        cfg.MODEL.DEVICE = 'cuda' if use_gpu else 'cpu'

        self.predictor = DefaultPredictor(cfg)

    def _preprocess(self, npz_path):

        # extract useful information from raw scan file, normalize the data and convert it to uint8
        CHANNELS = [("reflectivity", 0.5), ("reflectivity", 1.5), ("velocity", 0.5)]
        NORMALIZERS = {
                'reflectivity':              pltc.Normalize(vmin=  -5, vmax= 35),
                'velocity':                  pltc.Normalize(vmin= -15, vmax= 15),
                'spectrum_width':            pltc.Normalize(vmin=   0, vmax= 10),
                'differential_reflectivity': pltc.Normalize(vmin=  -4, vmax= 8),
                'differential_phase':        pltc.Normalize(vmin=   0, vmax= 250),
                'cross_correlation_ratio':   pltc.Normalize(vmin=   0, vmax= 1.1)
        }
        attributes = ['reflectivity', 'velocity', 'spectrum_width']
        elevations = [0.5, 1.5, 2.5, 3.5, 4.5]

        array = np.load(npz_path)["array"]
        image = np.stack([NORMALIZERS[attr](array[attributes.index(attr), elevations.index(elev), :, :])
                        for (attr, elev) in CHANNELS], axis=-1)
        np.nan_to_num(image, copy=False, nan=0.0)
        image = (image * 255).astype(np.uint8)
        
        return image

    def run(self, npz_files):
        
        outputs = []
        count = 0
        for idx, npz_file in enumerate(tqdm(npz_files, desc="Detecting")):

            # extract scanname 
            name = os.path.splitext(os.path.basename(npz_file))[0]
            # preprocess data
            data = self._preprocess(npz_file)
            # detect roosts 
            prediction = self.predictor(data)["instances"]
            scores     = prediction.scores.cpu().numpy()
            if len(scores) == 0: # no roost detected in this scan
                continue
            bbox       = prediction.pred_boxes.tensor.cpu().numpy()
            H, W       = prediction.image_size
            centers    = prediction.pred_boxes.get_centers().cpu().numpy()
            centers[:, 1] = H - centers[:, 1] # flip the y direction
            radius     = ((bbox[:, 2] - bbox[:, 0]) + (bbox[:, 3] - bbox[:, 1])) / 4.
            radius     = radius[:, np.newaxis]
            bbox_xyr   = np.hstack((centers, radius))
            # reformat the detections
            for kk in range(len(scores)):
                det = {"scanname" : name, 
                       "det_ID"   : count,
                       "det_score": scores[kk], # just in case we run detector on GPU
                       "im_bbox"  : bbox_xyr[kk]}
                count += 1
                outputs.append(det)
        return outputs


if __name__ == "__main__":
    
    # model 
    ckpt_path = "/home/zezhoucheng/random_repo/roosts/experiments/experiments/01-first-trial/logs/KDOX_lr_0.01000/model_final.pth"
    detector = Detector(ckpt_path)

    # data
    data_path = "/home/zezhoucheng/random_repo/roosts/libs/wsrdata/static/arrays/v0.1.0/2008/10/17/KDOX/KDOX20081017_112306_V04.npz"
    data_path2 = "/home/zezhoucheng/random_repo/roosts/libs/wsrdata/static/arrays/v0.1.0/2010/10/02/KDOX/KDOX20101002_112200_V04.npz"

    # make prediction
    outputs = detector.run([data_path, data_path2])
    print(outputs)
    

