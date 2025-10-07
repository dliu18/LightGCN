#!/bin/bash


TRIAL_NUM=$1

DATASET="gowalla"
EPOCHS=500
BATCH_SIZE=2048


LGN_ALPHA=0.5
LGN_BETA=0.0
LGN_IPW_BETA=-0.1
LGN_POP_CORR_LAMBDA=1
LGN_PC_ALPHA=0.000001
LGN_PC_BETA=0.0

MF_ALPHA=0.0
MF_BETA=-0.5
MF_IPW_BETA=-0.5
MF_POP_CORR_LAMBDA=1
MF_PC_ALPHA=0.000001
MF_PC_BETA=0.0

BASE_CMD="python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --epochs=$((${EPOCHS} + 1)) --bpr_batch=${BATCH_SIZE} \
--test_set=test --test_interval=${EPOCHS} --topks=\"[20, 10000]\" --recdim=64 --dataset=$DATASET --model=$MODEL \
--use_cpp=1 --sample_pos=1 --normalize_items=0 --normalize_users=0 --shuffle_users=1 --tau=0"

# Define trials
declare -a TRIALS=(

	#LGN vanilla
	"NAME=vanilla MODEL=lgn ALPHA=0 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#LGN vanilla BPR
	"NAME=vanilla-bpr MODEL=lgn ALPHA=1 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#LGN ours
	"NAME=ours MODEL=lgn ALPHA=${LGN_ALPHA} BETA=${LGN_BETA} POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#LGN ipw
	"NAME=ipw MODEL=lgn ALPHA=0 BETA=${LGN_IPW_BETA} POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#LGN pop_reg
	"NAME=pop_reg MODEL=lgn ALPHA=0 BETA=0 POP_CORR_LAMBDA=${LGN_POP_CORR_LAMBDA} PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#LGN pop_comp
	"NAME=vanilla MODEL=lgn ALPHA=0 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=${LGN_PC_ALPHA} PC_BETA=${LGN_PC_BETA} LOAD=1 OUTPUT_FILE=www-final/lgn/${DATASET}/pop_comp/${TRIAL_NUM}"

	#MF vanilla
	"NAME=vanilla MODEL=mf ALPHA=0 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#MF vanilla BPR
	"NAME=vanilla-bpr MODEL=mf ALPHA=1 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#MF ours
	"NAME=ours MODEL=mf ALPHA=${MF_ALPHA} BETA=${MF_BETA} POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#MF ipw
	"NAME=ipw MODEL=mf ALPHA=0 BETA=${MF_IPW_BETA} POP_CORR_LAMBDA=0 PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#MF pop_reg
	"NAME=pop_reg MODEL=mf ALPHA=0 BETA=0 POP_CORR_LAMBDA=${MF_POP_CORR_LAMBDA} PC_ALPHA=0 PC_BETA=0 LOAD=0 OUTPUT_FILE=''"

	#MF pop_comp
	"NAME=vanilla MODEL=mf ALPHA=0 BETA=0 POP_CORR_LAMBDA=0 PC_ALPHA=${MF_PC_ALPHA} PC_BETA=${MF_PC_BETA} LOAD=1 OUTPUT_FILE=www-final/mf/${DATASET}/pop_comp/${TRIAL_NUM}"

)

# Run trials
for TRIAL in "${TRIALS[@]}"; do
    # Extract individual parameters
    eval $TRIAL

	comment="www-final/${MODEL}/${DATASET}/${NAME}/${TRIAL_NUM}"

    CMD="$BASE_CMD --comment=\"${comment}\" \
    	--model=${MODEL} \
    	--alpha=${ALPHA} \
    	--beta=${BETA} \
    	--pop_corr_lambda=${POP_CORR_LAMBDA} \
    	--pc_alpha=${PC_ALPHA} \
    	--pc_beta=${PC_BETA} \
    	--load=${LOAD} \
    	--output_file=${OUTPUT_FILE}"

    echo "Running: $CMD"
    eval $CMD
done