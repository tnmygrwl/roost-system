import csv
import matplotlib.pyplot as plt
import numpy
import torch
import matplotlib as mpl
from matplotlib import cm
import os
from wsrlib import radar2mat
import pyart
from PIL import Image
from PIL import ImageDraw
import json

def loadScan_pyart(scan):
    radar = pyart.io.read_nexrad_archive(scan)
    
    fields = ['reflectivity', 'velocity', 'spectrum_width']
    #data, _,_, elev = radar2mat(radar, 
    data, fields, elev, _, _ = radar2mat(radar, 
                              fields=fields,
                              as_dict=False,
                              coords='cartesian',
                              elevs=[0.5,1,2.5,3.5,4.5],
                              dim=600,r_max=150000)
    print(fields)

    for f in range(0,len(fields)):
        data[f] = data[f][:,::-1,:].copy()
    render = [data[f] for f in range(0,len(fields))]
    print(len(render))
    print(data[0].shape)
    render = np.concatenate(render, axis=0)
    print(render.shape)
    # density = dz.copy()
    density = data[0] #0 is reflectivity as the pywsrlib uses a list of fiels by number
    density = idb(density)
    density[density != density] = 0
    density, _ = z_to_refl(density)

    ee, h, w = density.shape
    pad_size = (((h // 32) + 1) * 32 - h) // 2
    # add padding
    render = np.pad(render,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode='edge')
    density = np.pad(density, 
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode='edge')
#    matplotlib.image.imsave(render)

    return (
        torch.from_numpy(render.astype(np.float32)),
        torch.from_numpy(density.astype(np.float32)),
        elev,
        pad_size,
        data[0]
    )

def read_labels(filename):
    pass

def render_img(npz_file, labels_file, rain_file, filename, bounding_boxes_list):
    print(labels_file.shape)
    cmap = mpl.colors.ListedColormap(
    numpy.array([
                 [0, 0.5, 1, 1],
                 [1, 0.5, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 0, 1]])
        )
    dz_cmap = cm.get_cmap('jet', 70)
    sm = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=70),
            cmap=dz_cmap
        )
    rm_cmap = cm.get_cmap('jet', 100)
    rm = cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=100),
            cmap=rm_cmap
        )
    concat_list = []
    for elev in range(5):
        rgba = sm.to_rgba(
            npz_file[elev,:,:] + 33,
                            bytes=True
                    )
        rgba_birds = rgba.copy()
        rain = torch.from_numpy(rain_file.numpy()[0,elev,::-1,:].copy())
        rgba_rain_probability = rm.to_rgba(
            rain[:,:]*100,
                            bytes=True
                    )

        #Mist net labels 
        #0 is background
	#1 is biology
	#2 is rain
	# Green (0,255,0) is birds, Blue (0,0,255) is rain. Background is grey
        rgba_birds[labels_file[0,elev] == 1, 0] = 0
        rgba_birds[labels_file[0,elev] == 1, 1] = 255
        rgba_birds[labels_file[0,elev] == 1, 2] = 0

        rgba_birds[labels_file[0,elev] == 2, 0] = 0
        rgba_birds[labels_file[0,elev] == 2, 1] = 0
        rgba_birds[labels_file[0,elev] == 2, 2] = 255

        rgba_birds[labels_file[0,elev] == 0, :3] = 127

        #rgba_background[labels_file[0,elev]==2, :3] = 255
        rgba_birds = generate_bounding_box_images(rgba_birds, bounding_boxes_list)
        rgba = generate_bounding_box_images(rgba, bounding_boxes_list)
        concat_elev = numpy.concatenate(
                [rgba,
                rgba_birds,
                rgba_rain_probability],
                axis=0
        )
        concat_list.append(concat_elev)
    viz_array = numpy.concatenate(concat_list, axis=1)
    mpl.image.imsave(
        os.path.join(
            os.path.basename(filename).split('.')[0] + '.png'
            ),
        viz_array
        )

def read_npz_file(detection_file):
    inputs = numpy.load(detection_file)
    return inputs['array']

def generate_bounding_box_images(data, bboxes):
# {"id": 386, "scan_id": 790, "category_id": 0, "sequence_id": 3440, 
# "x": -9972.0, "y": 79972.0, "r": 7704.845465528156, "x_im": 280.056,
# "y_im": 459.944, "r_im": 15.409690931056309, 
# "bbox": [263.71938264926126, 443.71938264926126, 31.561234701477474, 31.561234701477474], "bbox_area": 996.1115358817459}
   # rgb = plt.imread(data)
   # plt.imshow(rgb)
    #print(data.shape)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.polygon([(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(bbox[0],bbox[1]+bbox[3])], outline='black')

    #"bbox": [272.2193463133217, 414.2193463133217, 6.5613073733566125, 6.5613073733566
    #draw.polygon([(272,414),(272+7,414),(272+7,414+7),(272,414+7)], outline='blue')
    new_data = numpy.asarray(img)
    #print(new_data.shape)
    if(numpy.array_equal(new_data,data)):
        print("somethign wrong")
    return new_data
    #x = float()
    #y = float(d[5])
    #r = float(d[6])
    #plt.axes()
    #rectangle = plt.Rectangle(263,443, 32, 32,fill=None,ec="red")
    #plt.gca().add_patch(rectangle)
    #plt.axis('scaled')
    #plt.show(block=False)
    #plt.pause(2)
    #plt.close()

def get_annos(json_annotations_file):
    files_to_process = []
    with open(json_annotations_file) as annos:
        json_data = json.load(annos)
        annotations = {x["id"]: x for x in json_data["annotations"]}
        for scan in json_data["scans"]:
            if "annotation_ids" in scan:
                if not scan["annotation_ids"] == []:
                    #print(scan)
                    #print(annotations[scan["annotation_ids"][0]])
                    files_to_process.append((scan["key"],scan['array_path'],[annotations[annotation]['bbox'] for annotation in scan["annotation_ids"]]))
    return files_to_process
                    
    return []
def main():
    json_annotations_file = "/scratch2/wenlongzhao/wsrdata/datasets/roosts_v0.1.0/roosts_v0.1.0.json"
    list_of_files_and_annotations = get_annos(json_annotations_file)
    print(len(list_of_files_and_annotations))

    for file in list_of_files_and_annotations:
        #print(file)
        if("KMLB19941212_120832" not in file):
            continue
        npz_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/" + file[1]
        labels_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/" + file[1].rsplit('/',1)[0] + "/labels_normalized/" + file[0] + ".pt"
        probs_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/" + file[1].rsplit('/',1)[0] + "/rain/" + file[0] + ".pt"
        #npz_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/2010/11/24/KAMX/KAMX20101124_113106_V03.npz"
        #labels_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/2010/11/24/KAMX/mistnet_labels/KAMX20101124_113106_V03.pt"
        #npz_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/2010/11/23/KAMX/KAMX20101123_113233_V03.npz"
        #labels_filename = "/scratch2/wenlongzhao/RadarNPZ/v0.1.0/2010/11/23/KAMX/mistnet_labels/KAMX20101123_113233_V03.pt"
        if os.path.isfile(labels_filename):
            image = render_img(read_npz_file(npz_filename)[0], torch.load(labels_filename), torch.load(probs_filename), npz_filename, file[2])
        else:
            #pass
            print("File not found:[" + labels_filename + "]")

if __name__ == "__main__":
    main()
