import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure

import os
from os.path import join, dirname
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
weight_file_folder = dirname(weight_file)
os.makedirs(weight_file_folder, exist_ok=True)

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    # if using the post-processing baseline, the board path is augmented
    w : SummaryWriter = SummaryWriter(world.BOARD_PATH)
else:
    w = None
    world.cprint("not enable tensorflowboard")

# try:
if world.LOAD:
    Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, w, world.config['multicore'])
else:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch > 0 and epoch % world.config["test_interval"] == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            # Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], is_test=False)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
# finally:
#     if world.tensorboard:
#         w.close()