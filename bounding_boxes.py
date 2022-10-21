import csv
import matplotlib.pyplot as plt
import numpy

def sort_dets(data):
    sorted_by_detection_id = {}
    for d in data:
        if d[0] in  sorted_by_detection_id:
            sorted_by_detection_id[d[0]].append(d)
        else:
            sorted_by_detection_id[d[0]] = [d]

    return sorted_by_detection_id


def read_dets(filename):
    with open("roosts_data/ui/scans_and_tracks/tracks_KTYX_20200812_20200812.txt") as csv:
        data = []
        for l in csv.readlines()[1:]:
            data.append(l.split(','))
    return data


def read_npz_file(detection_file):
    return str("roosts_data/ui/img/dz05/2020/08/12/KTYX/" + detection_file + ".png")
def generate_bounding_box_images(detection):

    detection.sort( key = lambda x: x[3])

    for d in detection:
        data = read_npz_file(d[1])
        rgb = plt.imread(data)
        plt.imshow(rgb)
        x = float(d[4])
        y = float(d[5])
        r = float(d[6])
        #plt.axes()
        rectangle = plt.Rectangle((x-r,y-r), 2*r, 2*r,fill=None,ec="red")
        plt.gca().add_patch(rectangle)
        plt.axis('scaled')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

def main():
    data = read_dets("roosts_data/ui/scans_and_tracks/tracks_KTYX_20200812_20200812.txt")

    print(data)
    sorted_data = sort_dets(data)
    print(sorted_data)
    
    for detection in sorted_data.keys():
        generate_bounding_box_images(sorted_data["17"])

if __name__ == "__main__":
    main()
