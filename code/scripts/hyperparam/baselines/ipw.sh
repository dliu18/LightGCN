#!/bin/bash

DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=${BATCH_SIZE} \
--test_set=valid --test_interval=${EPOCHS} --topks=\"[20, 20]\" --recdim=64 --dataset=pp/$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--alpha=0"

# Run trials
for BETA in -0.1 -0.25 -0.5 -1.0; do
	comment="hyperparam/baselines/${MODEL}/${DATASET}/ipw/${BETA}"

    CMD="${BASE_CMD} --beta=${BETA} --comment=\"${comment}\""

    echo "Running: $CMD"
    eval $CMD
done