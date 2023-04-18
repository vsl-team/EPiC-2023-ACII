#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

run_ver=$(date +%y%m%d_%H%M%S)

# Dataloader
data_dir="/home/hvthong/sXProject/EPiC23/data"
#scenario=1
#fold=0
val_ratio=0.

# Train
batch_size=128
num_epochs=10
step_per_epoch=2500

# Optimizer
opt_name='adamw'
base_lr=0.001
wd=0.01
warmup_steps=0.1
decay_steps=-1.
lr_sched='cosine'
alpha=0.0

# Model
seq_len=2000
n_stages=4
n_heads=4
dim_ff=64
n_layers=4
hid_mult=4

out_dir="/mnt/Work/Dataset/ACII-23/EPiC/train_logs/${run_ver}/scenario_${scenario}"

for scenario in 1 2 3 4; do
  for fold in 0 1 2 3 4; do
    echo "SCENARIO ${scenario} FOLD ${fold}"
    if [ $scenario == 1 ]; then
      out_dir="/mnt/Work/Dataset/ACII-23/EPiC/train_logs/${run_ver}/scenario_${scenario}/"
    else
      out_dir="/mnt/Work/Dataset/ACII-23/EPiC/train_logs/${run_ver}/scenario_${scenario}/fold_${fold}"
    fi
    TF_CPP_MIN_LOG_LEVEL=2 python -W ignore main.py --cfg config/base_cfg.yaml \
      OUT_DIR $out_dir DEBUG False \
      MODEL.SEQ_LEN $seq_len MODEL.N_STAGES $n_stages MODEL.N_LAYERS $n_layers MODEL.N_HEADS $n_heads \
      MODEL.DIM_FF $dim_ff MODEL.HID_MULT $hid_mult \
      DATA_LOADER.DATA_DIR $data_dir \
      DATA_LOADER.SCENARIO $scenario \
      DATA_LOADER.FOLD $fold \
      DATA_LOADER.VAL_RATIO $val_ratio \
      TRAIN.BATCH_SIZE $batch_size \
      TRAIN.EPOCHS $num_epochs \
      TRAIN.STEP_PER_EPOCH $step_per_epoch \
      OPTIM.BASE_LR $base_lr \
      OPTIM.NAME $opt_name \
      OPTIM.WEIGHT_DECAY $wd \
      OPTIM.WARMUP_STEPS $warmup_steps \
      OPTIM.LR_SCHEDULER_DECAY_STEPS $decay_steps \
      OPTIM.LR_SCHEDULER $lr_sched \
      OPTIM.LR_SCHEDULER_ALPHA $alpha

    sleep 1
    if [ $scenario == 1 ] && [ $fold == 0 ]; then
      break
    fi
    if [ $scenario == 3 ] && [ $fold == 3 ]; then
      break
    fi
    if [ $scenario == 4 ] && [ $fold == 1 ]; then
      break
    fi
  done
done
