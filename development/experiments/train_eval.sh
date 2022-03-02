GPU=${1:-0}
MODEL=${2:-"31"}
ADAPTOR=${3:-"linear"}
INPUT_CHANNELS=${4:-15}
IMSIZE=${5:-1100}
BACKBONE=${6:-"resnet101-FPN"}
ITER=${7:-50000}
DATASET=${8:-1}
TEST_DATASET=${9:-1}

#ROOT_DIR="/mnt/nfs/scratch1/gperezsarabi/darkecology"
ROOT_DIR="/scratch1/gperezsarabi/darkecology/"


CUDA_VISIBLE_DEVICES=$GPU python train_roost_detector.py --train_dataset $DATASET --imsize $IMSIZE --adaptor $ADAPTOR --input_channels $INPUT_CHANNELS --network $BACKBONE --anchor_strategy 10 --reg_loss smooth_l1 --lr 0.001 --max_iter $ITER --eval_period 0 --checkpoint_period 5000 --output_dir ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}

: '
CUDA_VISIBLE_DEVICES=$GPU python eval_roost_detector.py --ckpt_path ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/model_final.pth --eval_strategy 1 --test_dataset $TEST_DATASET --imsize $IMSIZE --adaptor $ADAPTOR --input_channels $INPUT_CHANNELS --network $BACKBONE --anchor_strategy 10 --output_dir ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/eval_ckpt$(($ITER - 1))_strt1

CUDA_VISIBLE_DEVICES=$GPU python eval_roost_detector.py --ckpt_path ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/model_00$(($ITER - 5001)).pth --eval_strategy 1 --test_dataset $TEST_DATASET --imsize $IMSIZE --adaptor $ADAPTOR --input_channels $INPUT_CHANNELS --network $BACKBONE --anchor_strategy 10 --output_dir ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/eval_ckpt$(($ITER - 5001))_strt1

CUDA_VISIBLE_DEVICES=$GPU python eval_roost_detector.py --ckpt_path ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/model_00$(($ITER - 10001)).pth --eval_strategy 1 --test_dataset $TEST_DATASET --imsize $IMSIZE --adaptor $ADAPTOR --input_channels $INPUT_CHANNELS --network $BACKBONE --anchor_strategy 10 --output_dir ${ROOT_DIR}/roost-system/development/experiments/$MODEL/logs/${BACKBONE}_detptr_anc10_regsl1_imsz${IMSIZE}_flip_lr0.001_it${ITER}/eval_ckpt$(($ITER - 10001))_strt1
'
