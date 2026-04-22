import time
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
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
        if epoch %100 == 0:
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
    print(Procedure.get_bpr_profile_summary())
    if torch.cuda.is_available() and world.device.type == "cuda":
        gpu_idx = world.device.index if world.device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_idx)
        peak_alloc = torch.cuda.max_memory_allocated(gpu_idx)
        peak_reserved = torch.cuda.max_memory_reserved(gpu_idx)
        total_mem = props.total_memory
        gib = 1024 ** 3
        print(
            "GPU memory summary | "
            f"device={gpu_idx} ({props.name}) | "
            f"peak_allocated={peak_alloc / gib:.2f} GiB | "
            f"peak_reserved={peak_reserved / gib:.2f} GiB | "
            f"total_available={total_mem / gib:.2f} GiB"
        )
    else:
        print("GPU memory summary | CUDA not available; training ran on CPU.")
    if world.tensorboard:
        w.close()
