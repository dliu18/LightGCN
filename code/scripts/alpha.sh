#!/bin/bash


DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4
BETA=$5

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=${BATCH_SIZE} \
--test_set=valid --test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --dataset=pp/$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_users=0 --normalize_items=0 --shuffle_users=1"

# Define trials
declare -a TRIALS=(
    "alpha=0"
    "alpha=0.25"
    "alpha=0.5"
    "alpha=0.75"
    "alpha=1.0"
    "alpha=1.5"
    "alpha=2.0"
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

