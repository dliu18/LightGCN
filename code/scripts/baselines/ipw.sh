#!/bin/bash

DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4
TRIAL_NUM=$5

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=2048 \
--test_set=test --test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--alpha=0"

# Run trials
for BETA in -0.1 -0.25 -0.5 -1.0; do
	comment="www/${MODEL}/${DATASET}/ipw/${BETA}/${TRIAL_NUM}"

    CMD="${BASE_CMD} --beta=${BETA} --comment=\"${comment}\""

    echo "Running: $CMD"
    eval $CMD
done