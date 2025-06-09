#!/bin/bash

# Usage: bash gowalla_full_eval.sh 

TRIAL=$1
DATASET="amazon-book"
MODEL="lgn"
LOAD=1

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=301 --bpr_batch=2048 \
--test_interval=300 --topks=\"[20, 2000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --shuffle_users=1 --tau=0"

# Define trials
declare -a TRIALS=(
    # #vanilla 
    "comment=final-${TRIAL}/${MODEL}/${DATASET}/vanilla normalize_users=0 beta=0 alpha=0 pc_beta=1.0 pc_alpha=0.00000000001"
)

# Run trials
for TRIAL in "${TRIALS[@]}"; do
    # Extract individual parameters
    eval $TRIAL

    CMD="$BASE_CMD --load=${LOAD} --alpha=${alpha} --beta=${beta} --pc_alpha=${pc_alpha} --pc_beta=${pc_beta} --normalize_users=${normalize_users} --comment=\"${comment}\""

    echo "Running: $CMD"
    eval $CMD
done
