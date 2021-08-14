import numpy as np
import os
import pdb
import sys

EXP_GROUP_NAME = "07"
# ROOT = f'/home/wenlongzhao/roosts-pytorch/experiments/{EXP_GROUP_NAME}'
ROOT = f'/mnt/nfs/work1/huiguan/wenlongzhao/roosts/experiments/{EXP_GROUP_NAME}'
CKPTS = range(49999, 55000, 5000)
EVAL_STRATEGY = 1 # ignore <15x15 in 1200x1200
EXP_DIRS = [
    "resnet101-FPN_detptr_anc6_regsl1_imsz1200_lr0.001_it150k",
    "resnet101-FPN_detptr_anc10_regsl1_imsz1200_lr0.001_it150k",
    # "resnet101-FPN_detptr_anc10_regsl1_imsz1200_lr0.005_it150k",
    "resnet101-FPN_detptr_anc10_reggiou_imsz1200_lr0.001_it150k",
    # "resnet101-FPN_detptr_anc10_regsl1_imsz1200_flip_lr0.001_it150k",
    # "resnet101-FPN_detptr_anc10_regsl1_imsz1200_rot_lr0.001_it150k",
    "resnet101-FPN_detptr_anc9_regsl1_imsz800_lr0.001_it150k",

    # "resnet50-c4_detptr_anc7_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_clsptr_anc4_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_clsptr_anc6_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_clsptr_anc8_imsz800_lr0.001_it150k",
    # "resnet50-FPN_detptr_anc4_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_detptr_anc6_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_detptr_anc8_imsz800_lr0.001_it150k",
    # "resnet50-FPN_noptr_anc4_imsz1200_lr0.001_it150k",
    # "resnet50-FPN_noptr_anc6_imsz1200_lr0.001_it150k",
]
EXCEPTION = [
    # (69999, "resnet50-FPN_noptr_anc6_imsz1200_lr0.001_it150k"),
]


script_dir = os.path.join(ROOT, 'scripts')
slurm_dir = os.path.join(ROOT, 'slurm')
log_dir = os.path.join(ROOT, 'logs')
os.makedirs(script_dir, exist_ok=True)
os.makedirs(slurm_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
launch_file = open(os.path.join(script_dir, 'launch_eval.sh'), 'w')
launch_file.write('#!/bin/bash\n')

exp_idx = 0
for exp_name in EXP_DIRS:
    network = exp_name.split("_")[0]
    imsize = exp_name.split("_imsz")[1].split("_")[0]
    anchor_strategy = exp_name.split("_anc")[1].split("_")[0]

    for i in CKPTS:
        if (i, exp_name) in EXCEPTION: continue
        ckpt_path = os.path.join(log_dir, exp_name, f"model_{i:07d}.pth")
        if os.path.exists(ckpt_path):
            eval_name = f"eval_{exp_name}_ckpt{i}_strt{EVAL_STRATEGY}"
            script_path = os.path.join(script_dir, eval_name+".sbatch")
            eval_name_brief = f"eval_ckpt{i}_strt{EVAL_STRATEGY}"
            output_dir = os.path.join(log_dir, exp_name, eval_name_brief)
            os.makedirs(output_dir, exist_ok=True)

            with open(script_path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('hostname\n')
                f.write(
                    ''.join((
                            f'python eval_roost_detector.py',
                            f' --ckpt_path {ckpt_path} --eval_strategy {EVAL_STRATEGY}',
                            f' --imsize {imsize}'
                            f' --network {network} --anchor_strategy {anchor_strategy}',
                            f' --output_dir {output_dir}',
                    ))
                )
            if exp_idx < 4:
                partition = "rtx8000-short"
            # elif exp_idx < 35:
            #     partition = "1080ti-long"
            elif exp_idx < 44:
                partition = "titanx-long"
            launch_file.write(
                f'sbatch -o {slurm_dir}/{eval_name}_%J.out '
                f'-p {partition} --exclude=node097,node122,node123 --gres=gpu:1 --mem=100000 {script_path}\n'
            )
            exp_idx += 1

