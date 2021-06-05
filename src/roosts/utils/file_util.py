import os
import os.path as osp
import shutil
from tqdm import tqdm

# not exist dir, then create
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# remove empty folders
def removeEmptyFolders(path, removeRoot=True):
    'Function to remove empty folders'
    if not os.path.isdir(path):
        return
    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)
    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        # print "Removing empty folder:", path
        os.rmdir(path)


# move files to a single dictionary for convinient.
def move_files(root):

    for path, dirlist, filelist in os.walk(root):
        if len(filelist) > 0:
            for name in filelist:
                filepath = osp.join(path, name)
                if not osp.exists(osp.join(root,name)) and osp.getsize(filepath) > 100:
                    # if the size of file is larger than a threshold
                    shutil.move(filepath, root)
                    # create an empty file with same name as a placehold
                    open(filepath, 'a').close() 

def delete_files(filepaths):
    for filepath in tqdm(filepaths, desc="Delete useless files"):
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print("The file does not exist")


if __name__ == '__main__':
    removeEmptyFolders('/home/zezhoucheng/czz-github/Ruling-the-Roost-with-CNNs/data/roost_demo_data/radar_scan/')
