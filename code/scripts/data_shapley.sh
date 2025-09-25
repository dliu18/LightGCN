#!/bin/bash

# ../LightGCN/data/amazon-book/power_niche/constant_users/0/


DATASET=$1
SAMPLE_TYPE=$2
EPOCHS=$3
NUM_TRIALS=$4

MODEL="lgn"

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=2048 \
--test_interval=${EPOCHS} --topks=\"[20, 2000]\" --recdim=64 --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--alpha=0 --beta=0"

# # # Run fixed dataset
CMD="$BASE_CMD \
--dataset=\"${DATASET}/fixed/${SAMPLE_TYPE}\" \
--comment=\"data_shapley/${DATASET}/fixed/${SAMPLE_TYPE}\""

echo "Running: $CMD"
eval $CMD

# Define trials
declare -a CONFIGS=(
	"dataset=${DATASET}/low_mainstream/${SAMPLE_TYPE}"
	"dataset=${DATASET}/low_niche/${SAMPLE_TYPE}"
	"dataset=${DATASET}/power_mainstream/${SAMPLE_TYPE}"
	"dataset=${DATASET}/power_niche/${SAMPLE_TYPE}"
	"dataset=${DATASET}/random-0.2/${SAMPLE_TYPE}"
)

# Run trials
for TRIAL_NUM in $(seq 0 $((NUM_TRIALS-1))); do
	for CONFIG in "${CONFIGS[@]}"; do
    	# Extract individual parameters
    	eval $CONFIG

	    CMD="$BASE_CMD \
	    --dataset=\"${dataset}/${TRIAL_NUM}\" \
	    --comment=\"data_shapley/${dataset}/${TRIAL_NUM}\""

	    echo "Running: $CMD"
	    eval $CMD
	done
done
