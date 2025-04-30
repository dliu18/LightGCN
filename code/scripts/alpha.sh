#!/bin/bash


DATASET=$1
MODEL=$2
EPOCHS=$3
BETA=$4

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$EPOCHS --bpr_batch=4096 \
--test_interval=150 --topks=\"[20, 2000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_users=0 --normalize_items=0 --shuffle_users=1"

# Define trials
declare -a TRIALS=(
    "alpha=0"
    "alpha=0.25"
    "alpha=0.5"
    "alpha=0.75"
    "alpha=1.0"
)

# Run trials
for TRIAL in "${TRIALS[@]}"; do
    # Extract individual parameters
    eval $TRIAL

    COMMENT="hyperparam/${MODEL}/${DATASET}/alpha/${alpha}/beta/${BETA}"

    CMD="$BASE_CMD --alpha=${alpha} --beta=${BETA} --comment=\"$COMMENT\""

    echo "Running: $CMD"
    eval $CMD
done

