import world
import utils
import torch
import numpy as np
from tqdm import tqdm 
import pickle

import register
from register import dataset

config = world.config
num_users = dataset.n_users
num_items = dataset.m_items

comment = "tau"
description = '''
    Embeddings for Gowalla trained with various values of tau. 
    The value of e-2 is equivalent to tau=infty 
    In all cases, degree-decay is e-2
    Only 1 item pair is chosen per user.
'''
predictions = {}
tau_str = {
    "0": "0.0",
    "2": "2.0",
    "4": "4.0",
    "8": "8.0",
    "e-2": "0.01"
}
for tau in ["0", "2", "4", "8", "e-2"]:
    Recmodel = register.MODELS[world.model_name](config, dataset)
    Recmodel = Recmodel.to(world.device)

    weight_file = "checkpoints/tau/{}/lgn-{}-{}-{}--{}.pth.tar".format(
        tau,
        world.dataset, 
        config["lightGCN_n_layers"], 
        config["latent_dim_rec"],
        tau_str[tau])
    Recmodel.load_state_dict(
        torch.load(
            weight_file,
            map_location=torch.device('cpu')
        )
    )

    # ratings = Recmodel.getUsersRating(torch.Tensor(range(num_users)))\
    #             .cpu()\
    #             .detach()\
    #             .numpy()

    user_embeddings, item_embeddings, _, _, _, _ = Recmodel.getEmbedding(
        torch.Tensor(range(num_users)).long().to("cuda"),
        torch.Tensor(range(num_items)).long().to("cuda"),
        torch.empty(0).long().to("cuda")
    ) 

    predictions[tau] = {}
    # predictions["ratings"] = ratings
    predictions[tau]["user embeddings"] = user_embeddings.cpu().detach().numpy()
    predictions[tau]["item embeddings"] = item_embeddings.cpu().detach().numpy()

    print(world.dataset)
    print(f"User embedding shape: {predictions[tau]['user embeddings'].shape}")
    print(f"Item embedding shape: {predictions[tau]['item embeddings'].shape}")


output_filename = "../../pickles/{}/lgn-predictions-{}.pickle".format(comment, world.dataset)
with open(output_filename, "wb") as pickleFile:
    pickle.dump((description, predictions), pickleFile)