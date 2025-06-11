#!/bin/bash

# Usage: bash gowalla_full_eval.sh 

TRIAL=$1
DATASET=$2
MODEL=$3

TEST_INTERVAL=$4
EPOCHS=$((TEST_INTERVAL + 1))

BATCH_SIZE=$5

LOAD=1

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=${EPOCHS} --bpr_batch=${BATCH_SIZE} \
--test_interval=${TEST_INTERVAL} --topks=\"[20, 2000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --shuffle_users=1 --tau=0 --load=1 \
--comment=\"final-${TRIAL}/${MODEL}/${DATASET}/vanilla\""

# alphas=($(seq 0.1 0.2 1.5))
alphas=(0.0001 0.001 0.01)

betas=($(seq 0 0.5 1.0))

for pc_alpha in "${alphas[@]}"; do
    for pc_beta in "${betas[@]}"; do

        CMD="$BASE_CMD --pc_alpha=${pc_alpha} --pc_beta=${pc_beta}"

        echo "Running: $CMD"
        eval $CMD
    done
done
