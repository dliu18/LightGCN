#!/bin/bash

# ../LightGCN/data/amazon-book/power_niche/constant_users/0/


DATASET=$1
SAMPLE_TYPE="constant_users"
EPOCHS=$2
BATCH_SIZE=$3
TRIAL_NUM=$4


MODEL="lgn"

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=${BATCH_SIZE} \
--test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--alpha=0 --beta=0"

for TREATMENT_RATIO in 0.01 0.05 0.1 0.3 0.5 0.9; do
	# # # Run fixed dataset
	CMD="$BASE_CMD \
	--dataset=\"${DATASET}-${TREATMENT_RATIO}/fixed/${SAMPLE_TYPE}\" \
	--comment=\"data_shapley/${DATASET}-${TREATMENT_RATIO}/fixed/${SAMPLE_TYPE}\""

	if [ ${TRIAL_NUM} = 0 ]; then
		echo "Running: $CMD"
		eval $CMD
	fi

	# Define trials
	declare -a CONFIGS=(
		"dataset=${DATASET}-${TREATMENT_RATIO}/low_mainstream/${SAMPLE_TYPE}"
		"dataset=${DATASET}-${TREATMENT_RATIO}/low_niche/${SAMPLE_TYPE}"
		"dataset=${DATASET}-${TREATMENT_RATIO}/power_mainstream/${SAMPLE_TYPE}"
		"dataset=${DATASET}-${TREATMENT_RATIO}/power_niche/${SAMPLE_TYPE}"
		"dataset=${DATASET}-${TREATMENT_RATIO}/random/${SAMPLE_TYPE}"
	)

	# Run trials
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
