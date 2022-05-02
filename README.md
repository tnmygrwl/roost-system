# Detecting and Tracking Communal Bird Roosts in Weather Radar Data
This repo implements a machine learning system for detecting and tracking communal bird roosts 
in weather surveillance radar data, continuing work by Cheng et al. [1].
Roost detection is based on [Detectron2](https://github.com/darkecology/detectron2) using PyTorch.

#### Repository Overview
- **checkpoints** is for trained model checkpoints
- **development** is for developing detection models
- **src** is for system implementation
    - **data**
        - **downloader** downloads radar scans based on station and day; 
        scan keys and directories for downloaded scans are based on UTC dates
        - **renderer** renders numpy arrays from downloaded scans, visualizes arrays, 
        and deletes the scans after rendering; 
        directories for rendered arrays and images are based on UTC dates
    - **detection**
    - **evaluation** contains customized evaluation adapted from pycocotools v2.0.2
    - **tracking**
    - **utils** contains various utils, scripts to postprocess roost tracks, and scripts to generate visualization
- **tools** is for system deployment
    - **demo.py** downloads radar scans, renders arrays, detects and tracks 
    roosts in them, and postprocesses the results 
    - **launch_demo.py** is a modifiable template that submits **demo.sbatch** to servers with slurm management
    - **demo.ipynb** is for interactively running the system
    - **utc_to_local_time.py** takes in web ui files and append local time to each line

#### Installation
1. Find a compatible PyTorch version 
[here](https://pytorch.org/get-started/previous-versions/) 
or [here](https://download.pytorch.org/whl/torch_stable.html).
To run detection with GPU, check the cuda version at, for example, `/usr/local/cuda`, or potentially by `nvcc -V`. 
    ```bash
    conda create -n roostsys python=3.6
    conda activate roostsys
    
    # for development and inference with gpus, use the gpu version of torch; we assume cuda 10.1 here
    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    # for inference with cpus, use the cpu version of torch
    # pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html
    
    git clone https://github.com/darkecology/roost-system.git
    cd roost-system
    pip install -e .
   ```

2. (Optional) Jupyter notebook.
- `pip install jupyter`
- Add the python environment to jupyter:
    ```bash
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=roostsys
    ```
- To check which environments are in jupyter as kernels and to delete one:
    ```bash
    jupyter kernelspec list
    jupyter kernelspec uninstall roostsys
    ```
- Run jupyter notebook on a server: `jupyter notebook --no-browser --port=9991`
- Monitor from local: `ssh -N -f -L localhost:9990:localhost:9991 username@server`
- Enter `localhost:9990` from a local browser tab

#### Developing a detection model
- **development** contains all training and evaluation scripts.
- To prepare a training dataset (i.e. rendering arrays from radar scans and 
generating json files to define datasets with annotations), refer to 
**Installation** and **Dataset Preparation** in the README of 
[wsrdata](https://github.com/darkecology/wsrdata.git).
- Before training, run **try_load_arrays.py** to make sure there's no broken npz files.

#### Run Inference
Inference can be run using CPU-only servers.
1. Under **checkpoints**, download a trained detection checkpoint.

2. [Configure AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) by
`aws configure`
in order to download radar scans. 
Enter `AWS Access Key ID` and `AWS Secret Access Key` as prompted,
`us-east-1` for `Default region name`, and nothing for `Default output format`.
Review the updated AWS config.
    ```bash
    vim ~/.aws/credentials
    vim ~/.aws/config
    ```

3. Modify **demo.py** for system customization. 
For example, DET_CFG can be changed to adopt a new detector.

4. In **tools**, modify VARIABLES in **launch_demo.py**. 
    1. EXPERIMENT_NAME needs to be carefully chosen; 
    it'll correspond to the dataset name later used in the website.
    2. If there are previous batches processed for this EXPERIMENT_NAME 
    (i.e. dataset to be loaded to the website),
    we can move previously processed data at the output directory to another location. 
    Then we can save the newly processed data at the same output directory; when we 
    copy the new data to the server hosting the website, 
    previous data don't need to be copied again.

5. In **tools**, run `python launch_demo.py` 
to submit jobs to slurm and process multiple batches of data.

#### Deployment Log
- v1: Beginning of Summer 2021 Zezhou model.
- v2: End of Summer 2021 Wenlong model with 48 AP. Good backbone, anchors, etc.
- v3: End of Winter 2021 Gustavo model with 55 AP. Adapter layer and temporal features.

#### Website Visualization
In the generated csv files, the following information could be used to further filter the tracks: 
- track length
- detection scores (-1 represents the bbox is not from detector, instead, our tracking algorithm)
- bbox sizes
- the minutes from sunrise/sunset of the first bbox in a track

#### Reference
[1] [Detecting and Tracking Communal Bird Roosts in Weather Radar Data.](https://people.cs.umass.edu/~zezhoucheng/roosts/radar-roosts-aaai20.pdf)
Zezhou Cheng, Saadia Gabriel, Pankaj Bhambhani, Daniel Sheldon, Subhransu Maji, Andrew Laughlin and David Winkler.
AAAI, 2020 (oral presentation, AI for Social Impact Track).
