#!/bin/bash

# FIGURE OUT LOADING OF VANILLA MODEL

DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4
TRIAL_NUM=$5

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=2048 \
--test_set=test --test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--beta=0 --alpha=0"

# Run trials
for PC_LAMBDA in 1 10 100 1000; do
	comment="www/${MODEL}/${DATASET}/pop_reg/${LAMBDA}/${TRIAL_NUM}"

    CMD="${BASE_CMD} --pop_corr_lambda=${LAMBDA} --comment=\"${comment}\""

    echo "Running: $CMD"
    eval $CMD
done