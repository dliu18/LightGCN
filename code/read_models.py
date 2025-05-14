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

dims = [2**i for i in range(1, 11)]

# dims = np.arange(10, 300, 10)
# dims = np.concatenate(([2, 5], dims))

predictions = {}
for dim in tqdm(dims):
    config["latent_dim_rec"] = dim
    Recmodel = register.MODELS[world.model_name](config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    weight_file = "checkpoints/pca-to-lgn/lgn-{}-bpr-{}-{}-0.5.pth.tar".format(
    	world.dataset, 
    	config["lightGCN_n_layers"], 
    	dim)
    Recmodel.load_state_dict(
        torch.load(
            weight_file,
            map_location=torch.device('cpu')
        )
    )
    
    ratings = Recmodel.getUsersRating(torch.Tensor(range(num_users)))\
                .cpu()\
                .detach()\
                .numpy()

    user_embeddings, item_embeddings, _, _, _, _ = Recmodel.getEmbedding(
        torch.Tensor(range(num_users)).long().to("cuda"),
        torch.Tensor(range(num_items)).long().to("cuda"),
        torch.empty(0).long().to("cuda")
    ) 
    
    predictions[dim] = {
        "ratings": ratings,
        "user embeddings": user_embeddings,
        "item embeddings": item_embeddings
    }
    
    output_filename = "../../pickles/lgn-predictions-{}.pickle".format(world.dataset)
    with open(output_filename, "wb") as pickleFile:
        pickle.dump(predictions, pickleFile)

    
