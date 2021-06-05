# Detecting and Tracking Communal Bird Roosts in Weather Radar Data
This repo implements a machine learning system for detecting and tracking communal bird roosts 
in weather surveillance radar data. It is a continuation of work by Cheng et al. [1].
Roost detection is based on [Detectron2](https://github.com/darkecology/detectron2) using PyTorch.

#### Under Development
- [ ] log
- [ ] dir structure, rsync results from swarm to doppler
- [ ] flip images during rendering, and also flip the y axis in the bounding box
       - [ ] renderer.py
       - [ ] utils/postprocess.py
- [ ] downloaded scans may not match the request. 
For example, {"station": "KDOX", "date": ("20101001", "20101001")} will download KDOX20101002 scans
- [ ] deployment: greatlake; scans with the old model; a year-station pair

#### Repository Overview
- **checkpoints** is for trained model checkpoints
- **libs** is for external libraries and toolkits, e.g. detectron2, wsrlib, wsrdata
- **src** is for system implementation
    - **data**
        - **downloader** downloads radar scans based on station and day
        - **renderer** renders numpy arrays from downloaded scans, visualizes arrays, 
        and deletes the scans after rendering
    - **detection**
    - **tracking**
    - **utils** contains various utils, scripts to postprocess roost tracks, and scripts to generate visualization
- **tools** is for system deployment
    - **demo.py** is a modifiable template for downloading radar scans, rendering arrays, and detecting and tracking 
    roosts in them. The detection model path and the system output directory should be specified in the file.

#### Installation and Preparation
1. Create and activate a python 3.6 environment. 
Install compatible [PyTorch](https://pytorch.org/get-started/previous-versions/) and OpenCV.
For GPU, check the cuda version at, for example, `/usr/local/cuda`, or potentially by `nvcc -V`. 
    ```bash
    conda create -n roost2021 python=3.6
    conda activate roost2021
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    # pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install opencv-python
    ```

2. Install this repo and its dependencies. 
    ```bash
    git clone https://github.com/darkecology/roost-system.git
    cd roost-system/libs
    
    # pywsrlib, commit 3690123 is tested
    git clone https://github.com/darkecology/pywsrlib.git
    pip install -e pywsrlib
    
    # detectron2, commit df6eac1 is tested
    git clone https://github.com/darkecology/detectron2.git
    pip install -e detectron2
    
    cd ..
    # conda install -c conda-forge scikit-learn  # fast nearest neightbor searching using ball tree for wind farm dataset
    pip install -e .
    ```

3. (Optional) [wsrdata](https://github.com/darkecology/wsrdata) is not needed for system deployment. 
It is only for preparing new roost datasets with annotations to train and evaluate new models.
    ```bash
    # wsrdata, commit 7ac8005 is tested
    # git clone https://github.com/darkecology/wsrdata.git
    # pip install -e wsrdata
    ``` 

4. Download the trained detection checkpoint `entire_lr_0.001.pth` from 
[here](https://www.dropbox.com/sh/g1m6406m6e087s6/AAAQyut5_ZF12zZ88tNiRzX-a?dl=0)
and place it under **checkpoints**.

5. [Configure AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) by
`aws configure`
in order to download radar scans. 
Enter `AWS Access Key ID` and `AWS Secret Access Key` as prompted,
`us-east-1` for `Default region name`, and nothing for `Default output format`.
Review the updated AWS config.
    ```bash
    vim ~/.aws/credentials
    vim ~/.aws/config
    ```

#### Reference
[1] [Detecting and Tracking Communal Bird Roosts in Weather Radar Data.](https://people.cs.umass.edu/~zezhoucheng/roosts/radar-roosts-aaai20.pdf)
Zezhou Cheng, Saadia Gabriel, Pankaj Bhambhani, Daniel Sheldon, Subhransu Maji, Andrew Laughlin and David Winkler.
AAAI, 2020 (oral presentation, AI for Social Impact Track).