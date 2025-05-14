import world
import torch
import register
import numpy as np
from register import dataset
from tqdm import tqdm 
import pickle

config = world.config
config["latent_dim_rec"] = 10

num_users = dataset.n_users
num_items = dataset.m_items

weight_files = [
    "lgn-lastfm-small-l2-0-10-0.0.pth.tar",
    "lgn-lastfm-small-l2-0-10-0.5.pth.tar",
    "lgn-lastfm-small-l2-1-10-0.5.pth.tar",
    "lgn-lastfm-small-l2-2-10-0.5.pth.tar",
    "lgn-lastfm-small-l2-3-10-0.5.pth.tar",
    "lgn-lastfm-small-l2-4-10-0.5.pth.tar",
    "lgn-lastfm-small-l2-5-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-0-10-0.0.pth.tar",
    "lgn-lastfm-small-bpr-0-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-1-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-2-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-3-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-4-10-0.5.pth.tar",
    "lgn-lastfm-small-bpr-5-10-0.5.pth.tar",
]

predictions = {}

for weight_file in tqdm(weight_files):
    Recmodel = register.MODELS[world.model_name](config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    Recmodel.load_state_dict(
        torch.load(
            "checkpoints/pca-to-lgn/" + weight_file,
            map_location=torch.device('cpu')
        )
    )
    
    print(weight_file)
    attr = weight_file.split("-")
    Recmodel.n_layers = int(attr[4])
    Recmodel.lam = float(attr[6][:2])
#     print(Recmodel.embedding_user.weight[0])

    ratings = Recmodel.getUsersRating(torch.Tensor(range(num_users)))\
                .cpu()\
                .detach()\
                .numpy()

    user_embeddings, item_embeddings, _, _, _, _ = Recmodel.getEmbedding(
        torch.Tensor(range(num_users)).long().to("cuda"),
        torch.Tensor(range(num_items)).long().to("cuda"),
        torch.empty(0).long().to("cuda")
    ) 
    
    predictions[weight_file] = {
        "ratings": ratings,
        "user embeddings": user_embeddings,
        "item embeddings": item_embeddings
    }
#     print(item_embeddings[0])
    
    with open("checkpoints/predictions-lastfm-small-pca-to-lgn.pickle", "wb") as pickleFile:
        pickle.dump(predictions, pickleFile)

    
