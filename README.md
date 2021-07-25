# Detecting and Tracking Communal Bird Roosts in Weather Radar Data
This repo implements a machine learning system for detecting and tracking communal bird roosts 
in weather surveillance radar data, continuing work by Cheng et al. [1].
Roost detection is based on [Detectron2](https://github.com/darkecology/detectron2) using PyTorch.

#### Repository Overview
- **checkpoints** is for trained model checkpoints
- **src** is for system implementation
    - **data**
        - **downloader** downloads radar scans based on station and day; 
        scan keys and directories for downloaded scans are based on UTC dates
        - **renderer** renders numpy arrays from downloaded scans, visualizes arrays, 
        and deletes the scans after rendering; 
        directories for rendered arrays are based on local dates
    - **detection**
    - **tracking**
    - **utils** contains various utils, scripts to postprocess roost tracks, and scripts to generate visualization
- **tools** is for system deployment
    - **demo.py** downloads radar scans, renders arrays, detects and tracks 
    roosts in them, and postprocesses the results 
    - **launch_demo.py** is a modifiable template that submits **demo.sbatch** to servers with slurm management
    - **demo.ipynb** is for interactively running the system

#### Installation and Preparation
1. Installation. Compatible PyTorch version can be found [here](https://pytorch.org/get-started/previous-versions/) 
and [here](https://download.pytorch.org/whl/torch_stable.html).
For running detection with GPU, check the cuda version at, for example, `/usr/local/cuda`, or potentially by `nvcc -V`. 
    ```bash
    conda create -n roost2021 python=3.6
    conda activate roost2021
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html
    # pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    git clone https://github.com/darkecology/roost-system.git
    cd roost-system
    pip install -e .
   ```

2. Under **checkpoints**, download a trained detection checkpoint.
    ```bash
    gdown --id 1oeGhJ7Bm2Uv3-BmIRhVsoVoTP5crOoem
    ```

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

4. Jupyter notebook.
- Install jupyter: `pip install jupyter`
- Add the python environment to jupyter:
    ```bash
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=roost2021
    ```
- To check which environments are in jupyter as kernels and to delete one:
    ```bash
    jupyter kernelspec list
    jupyter kernelspec uninstall roost2021
    ```
- Run jupyter notebook on a server: `jupyter notebook --no-browser --port=9991`
- Monitor from local: `ssh -N -f -L localhost:9990:localhost:9991 username@server`
- Enter `localhost:9990` from a local browser tab

#### Run Inference
In **tools**, modify **launch_demo.py** and run `python launch_demo.py` 
to submit jobs to slurm and process multiple batches of data. 
For additional customization, modify **demo.py**.
Currently the system renders more channels than needed for detection and tracking; 
inference could be accelerated by rendering only useful channels.

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
