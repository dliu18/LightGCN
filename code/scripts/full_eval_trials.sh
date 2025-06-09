#!/bin/bash

# Usage: bash gowalla_full_eval.sh 

DATASET=$1
MODEL=$2

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=301 --bpr_batch=2048 \
--test_interval=300 --topks=\"[20, 2000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --shuffle_users=1 --tau=0"

for TRIAL_INDEX in 1 2 3; do

	# Define trials
	declare -a TRIALS=(
		# #vanilla 
		# "comment=final-${TRIAL_INDEX}/${MODEL}/${DATASET}/vanilla normalize_users=0 beta=0 alpha=0"

		# #optimal recall from hyperparam
		"comment=final-${TRIAL_INDEX}/${MODEL}/${DATASET}/ours normalize_users=0 beta=-0.5 alpha=1.0"

		# #optimal recall from hyperparam where alpha=0 and beta is not zero (only item norm)
		# "comment=final-${TRIAL_INDEX}/${MODEL}/${DATASET}/only-item normalize_users=0 beta=-0.5 alpha=0"

		#vanilla but normalize users
		# "comment=final-${TRIAL_INDEX}/${MODEL}/${DATASET}/only-users normalize_users=1 beta=0 alpha=1"

	)

	# Run trials
	for TRIAL in "${TRIALS[@]}"; do
	    # Extract individual parameters
	    eval $TRIAL

	    CMD="$BASE_CMD --alpha=${alpha} --beta=${beta} --normalize_users=${normalize_users} --comment=\"${comment}\""

	    echo "Running: $CMD"
	    eval $CMD
	done
done

# python main.py --decay=1e-4 --lr=0.001 --layer=3 --bpr_batch=4096 --seed=2020 --epochs=701 --test_interval=100 --topks="[20, 2000]" --recdim=64 --use_cpp=1 --dataset="amazon-book" --sample_pos=1 --normalize_users=0 --normalize_items=0 --shuffle_users=1 --alpha=0.0 --tau=0 --comment="sanity-check/amazon-book-large-batch"