#!/bin/bash

# top-k is set at 20 because we are not optimizing for popularity bias during hyperparameter optimizaton

DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=${BATCH_SIZE} \
--test_set=valid --test_interval=${EPOCHS} --topks=\"[20, 20]\" --recdim=64 --dataset=pp/$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 \
--beta=0 --alpha=0 \
--load=1"

# Run trials
# for PC_ALPHA in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
for PC_ALPHA in 0.01 0.001 0.0001 0.00001 0.000001; do
    for PC_BETA in 0 0.25 0.5 0.75 1.0; do
        comment="hyperparam/${MODEL}/${DATASET}/alpha/0/beta/0"

    	output_file="hyperparam/baselines/${MODEL}/${DATASET}/pop_comp/${PC_ALPHA}/${PC_BETA}"

        CMD="${BASE_CMD} --pc_alpha=${PC_ALPHA} --pc_beta=${PC_BETA} --comment=\"${comment}\" --output_file=\"${output_file}\""

        echo "Running: $CMD"
        eval $CMD
    done
done