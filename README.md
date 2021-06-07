# Detecting and Tracking Communal Bird Roosts in Weather Radar Data
This repo implements a machine learning system for detecting and tracking communal bird roosts 
in weather surveillance radar data, continuing work by Cheng et al. [1].
Roost detection is based on [Detectron2](https://github.com/darkecology/detectron2) using PyTorch.

#### Under Development
- [ ] log
- [ ] dir structure, rsync results from swarm to doppler
- [ ] image and bbox directions: utils/postprocess.py
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
1. Installation. Compatible PyTorch version can be found [here](https://pytorch.org/get-started/previous-versions/).
For running detection with GPU, check the cuda version at, for example, `/usr/local/cuda`, or potentially by `nvcc -V`. 
    ```bash
    conda create -n roost2021 python=3.6
    conda activate roost2021
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    git clone https://github.com/darkecology/roost-system.git
    pip install -e .
   ```

2. Download the trained detection checkpoint `entire_lr_0.001.pth` from 
[here](https://www.dropbox.com/sh/g1m6406m6e087s6/AAAQyut5_ZF12zZ88tNiRzX-a?dl=0)
and place it under **checkpoints**.

3. [Configure AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) by
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