#!/bin/bash

# Usage: bash tune.sh gowalla

DATASET=$1
BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=600 --topks=\"[20, 2000]\" --recdim=64 --use_cpp=1 --dataset=$DATASET"

# Define trials
declare -a TRIALS=(
    # "sample_pos=1 normalize_users=0 normalize_items=0 tau=0"
    # "sample_pos=1 normalize_users=0 normalize_items=1 tau=0"
    # "sample_pos=1 normalize_users=0 normalize_items=1 tau=1"
    # "sample_pos=0 normalize_users=0 normalize_items=0 tau=0"
    # "sample_pos=0 normalize_users=0 normalize_items=1 tau=0"
    # "sample_pos=0 normalize_users=0 normalize_items=1 tau=1"

    "sample_pos=1 normalize_users=0 normalize_items=0 tau=0"
    # "sample_pos=0 normalize_users=0 normalize_items=0 tau=0"
    # "sample_pos=0 normalize_users=1 normalize_items=0 tau=0"
    # "sample_pos=0 normalize_users=0 normalize_items=1 tau=0"

)

# Run trials
for TRIAL in "${TRIALS[@]}"; do
    # Extract individual parameters
    eval $TRIAL

    COMMENT="temp/${DATASET}/sample_pos/${sample_pos}/normalize_items/${normalize_items}/normalize_users/${normalize_users}/tau/${tau}"

    CMD="$BASE_CMD --sample_pos=${sample_pos} --normalize_users=${normalize_users} --normalize_items=${normalize_items} --tau=${tau} --shuffle_users=1 --comment=\"$COMMENT\""

    echo "Running: $CMD"
    eval $CMD
done