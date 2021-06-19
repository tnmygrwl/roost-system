import os
import os.path as osp
import shutil
from tqdm import tqdm

# not exist dir, then create
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_files(filepaths):
    # for filepath in tqdm(filepaths, desc="Delete useless files"):
    for filepath in filepaths:
        if os.path.exists(filepath):
            os.remove(filepath)

