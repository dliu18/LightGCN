#!/bin/bash

# ./scripts/full_eval.sh amazon-book lgn 100 4096 0.25 -0.25 0

DATASET=$1
MODEL=$2
EPOCHS=$3
BATCH_SIZE=$4
ALPHA=$5
BETA=$6
TRIAL_NUM=$7

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=2048 \
--test_set=test --test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --shuffle_users=1 --tau=0"

# Define trials
declare -a TRIALS=(
	# #vanilla 
	# "comment=www/${MODEL}/${DATASET}/vanilla/${TRIAL_NUM} normalize_users=0 beta=0 alpha=0"

	# #optimal recall from hyperparam
	"comment=www/${MODEL}/${DATASET}/ours/${TRIAL_NUM} normalize_users=0 beta=${BETA} alpha=${ALPHA}"
)

# Run trials
for TRIAL in "${TRIALS[@]}"; do
    # Extract individual parameters
    eval $TRIAL

    CMD="$BASE_CMD --alpha=${alpha} --beta=${beta} --normalize_users=${normalize_users} --comment=\"${comment}\""

    echo "Running: $CMD"
    eval $CMD
done