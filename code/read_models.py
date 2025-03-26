import world
import torch
import register
import numpy as np
from register import dataset
from tqdm import tqdm 
import pickle

config = world.config
num_users = dataset.n_users
num_items = dataset.m_items


Recmodel = register.MODELS[world.model_name](config, dataset)
Recmodel = Recmodel.to(world.device)

weight_file = "checkpoints/lgn-{}-{}-{}.pth.tar".format(
    world.dataset, 
    config["lightGCN_n_layers"], 
    config["latent_dim_rec"])
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

predictions = {}
# predictions["ratings"] = ratings
predictions["user embeddings"] = user_embeddings.cpu().detach().numpy()
predictions["item embeddings"] = item_embeddings.cpu().detach().numpy()

print(world.dataset)
print(f"User embedding shape: {predictions['user embeddings'].shape}")
print(f"Item embedding shape: {predictions['item embeddings'].shape}")


output_filename = "../../pickles/lgn-predictions-{}.pickle".format(world.dataset)
with open(output_filename, "wb") as pickleFile:
    pickle.dump(predictions, pickleFile)