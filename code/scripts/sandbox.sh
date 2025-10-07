#!/bin/bash

# DATASET="amazon-book"
# MODEL="lgn"
# COMMENT="debug/temp"

# CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=0 --bpr_batch=2048 \
# --dataset=${DATASET} --comment=${COMMENT} \
# --test_set=test --test_interval=1 --topks=\"[20, 10000]\" --recdim=64 --model=${MODEL} \
# --use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
# --alpha=0 --beta=0 \
# --pop_corr_lambda=10.0"

# eval $CMD

# for TRIAL_NUM in 0 1 2 3; do
# 	CMD="./scripts/data_shapley_by_treatment_ratio.sh gowalla 500 2048 ${TRIAL_NUM}"
# 	eval $CMD
# done 

# for TRIAL_NUM in 0 1 2 3 4; do
# 	CMD="./scripts/data_shapley_by_treatment_ratio.sh yelp2018 300 2048 ${TRIAL_NUM}"
# 	eval $CMD
# done 

for TRIAL_NUM in 0 1 2 3; do
	CMD="./scripts/data_shapley_by_treatment_ratio.sh amazon-book 100 4096 ${TRIAL_NUM}"
	eval $CMD
done 