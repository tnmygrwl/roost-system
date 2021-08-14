import numpy as np
import os
import pdb
import sys

EXP_GROUP_NAME = '07'
# ROOT = f'/mnt/nfs/work1/smaji/wenlongzhao/roosts/experiments/{EXP_GROUP_NAME}'
ROOT = f'/mnt/nfs/work1/huiguan/wenlongzhao/roosts/experiments/{EXP_GROUP_NAME}'
MAX_ITER = 150000
CKPT_PERIOD = 5000

NET_IMSZ_ANCHOR_REGLOSS = [
    # ("resnet50-FPN", 1200, 4),
    # ("resnet50-FPN", 1200, 6),
    # ("resnet50-FPN", 800, 8),
    # ("resnet50-c4", 1200, 7),
    ("resnet101-FPN", 1200, 6, "smooth_l1"),
    ("resnet101-FPN", 1200, 6, "giou"),
    ("resnet101-FPN", 1200, 10, "smooth_l1"),
    ("resnet101-FPN", 1200, 10, "giou"),
    ("resnet101-FPN", 800, 9, "smooth_l1")
]
PRETRAIN_LR = [
    ("det", 0.001),
    # ("det", 0.005),
    # ("det", 0.01),
    # ("cls", 0.001),
    # ("cls", 0.005),
    # ("cls", 0.01),
    # ("no", 0.001),
    # ("no", 0.01),
    # ("no", 0.02),
]
FLIP_AND_ROTATE = [
    (True, False),
]
FILTER_EMPTY = [False] # [True, False]


script_dir = os.path.join(ROOT, 'scripts')
slurm_dir = os.path.join(ROOT, 'slurm')
log_dir = os.path.join(ROOT, 'logs')
os.makedirs(script_dir, exist_ok=True)
os.makedirs(slurm_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
launch_file = open(os.path.join(script_dir, 'launch_train.sh'), 'w')
launch_file.write('#!/bin/bash\n')

exp_idx = 0
for (network, imsize, anchor_strategy, reg_loss) in NET_IMSZ_ANCHOR_REGLOSS:
    for (pretrain, lr) in PRETRAIN_LR:
        for (flip, rotate) in FLIP_AND_ROTATE:
            for filter_empty in FILTER_EMPTY:
                exp_name = f"{network}_{pretrain}ptr_anc{anchor_strategy}" \
                           f"_reg{'giou' if reg_loss=='giou' else 'sl1'}" \
                           f"_imsz{imsize}" \
                           f"{'_flip' if flip else ''}{'_rot' if rotate else ''}" \
                           f"{'_flt' if filter_empty else ''}" \
                           f"_lr{lr:.3f}_it{MAX_ITER//1000}k"
                script_path = os.path.join(script_dir, exp_name+'.sbatch')
                output_dir = os.path.join(log_dir, exp_name)
                os.makedirs(output_dir, exist_ok=True)

                with open(script_path, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('hostname\n')
                    f.write(
                        ''.join((
                            f'python train_roost_detector.py',
                            f' --imsize {imsize}',
                            f' --flip' if flip else '',
                            f' --rotate' if rotate else '',
                            f' --filter_empty' if filter_empty else '',
                            f' --network {network} --pretrain {pretrain}'
                            f' --anchor_strategy {anchor_strategy} --reg_loss {reg_loss}',
                            f' --lr {lr} --max_iter {MAX_ITER}',
                            f' --checkpoint_period {CKPT_PERIOD} --output_dir {output_dir}',
                        ))
                    )
                partition = "1080ti-long" if exp_idx < 40 else "titanx-long"
                launch_file.write(
                    f'sbatch -o {slurm_dir}/{exp_name}_%J.out '
                    f'-p {partition} --exclude=node097,node119,node122,node123 --gres=gpu:1 --mem=100000 {script_path}\n'
                )
                exp_idx += 1
