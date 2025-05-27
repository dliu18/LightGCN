#!/bin/bash

# Initial values for the arguments
args=(2 4 8 16 32 64 128 256 512 1024)

dataset=$1

# Loop through each argument and call the Python script
for arg in "${args[@]}"; do
    python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --epochs=125 --dataset="$dataset" --topks="[20]" --recdim="$arg"
done

python read_models.py --layer=2 --seed=2020 --dataset="$dataset"
