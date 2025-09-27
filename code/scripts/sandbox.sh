#!/bin/bash

DATASET="gowalla"
COMMENT="debug"

CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=501 --bpr_batch=2048 \
--dataset=${DATASET} --comment=${COMMENT} \
--test_set=test --test_interval=500 --topks=\"[20, 2000]\" --recdim=64 --model=lgn \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0 \
--alpha=0 --beta=0"

eval $CMD