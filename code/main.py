import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import group_mixing
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
mix_state = group_mixing.build_group_mixing_state(dataset.n_users)
allowed_eval_users = None
if mix_state is not None:
    allowed_eval_users = mix_state["source_users"]

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(
                dataset,
                Recmodel,
                epoch,
                w,
                world.config['multicore'],
                allowed_users=allowed_eval_users,
            )
        output_information = Procedure.BPR_train_original(
            dataset,
            Recmodel,
            bpr,
            epoch,
            neg_k=Neg_k,
            w=w,
            mix_state=mix_state,
        )
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    cprint("[FINAL TEST]")
    Procedure.Test(
        dataset,
        Recmodel,
        world.TRAIN_epochs,
        w,
        world.config['multicore'],
        allowed_users=allowed_eval_users,
    )
finally:
    if world.tensorboard:
        w.close()
