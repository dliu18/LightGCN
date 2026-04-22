'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from collections import defaultdict
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2
_BPR_PROFILE_TOTALS = defaultdict(float)
_BPR_PROFILE_EPOCHS = 0


def _sync_cuda_if_needed():
    if world.device.type == "cuda":
        torch.cuda.synchronize(world.device)


def _format_profile_seconds(profile):
    keys = [
        "sample",
        "tensorize",
        "to_device",
        "shuffle",
        "weight_lookup",
        "stage_one",
        "tb_log",
        "batch_loop",
        "epoch_total",
    ]
    parts = []
    for key in keys:
        if key in profile:
            parts.append(f"{key}:{profile[key]:.4f}s")
    return "{" + ", ".join(parts) + "}"


def get_bpr_profile_summary(reset=False):
    global _BPR_PROFILE_TOTALS, _BPR_PROFILE_EPOCHS
    if _BPR_PROFILE_EPOCHS == 0:
        summary = "BPR runtime profile (cumulative): no epochs profiled."
    else:
        summary = (
            f"BPR runtime profile (cumulative over {_BPR_PROFILE_EPOCHS} epochs): "
            f"{_format_profile_seconds(_BPR_PROFILE_TOTALS)}"
        )
    if reset:
        _BPR_PROFILE_TOTALS = defaultdict(float)
        _BPR_PROFILE_EPOCHS = 0
    return summary


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None, mix_state=None):
    global _BPR_PROFILE_TOTALS, _BPR_PROFILE_EPOCHS
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    epoch_profile = defaultdict(float)
    _sync_cuda_if_needed()
    epoch_t0 = time()

    _sync_cuda_if_needed()
    t0 = time()
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    _sync_cuda_if_needed()
    epoch_profile["sample"] += time() - t0

    _sync_cuda_if_needed()
    t0 = time()
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    _sync_cuda_if_needed()
    epoch_profile["tensorize"] += time() - t0

    _sync_cuda_if_needed()
    t0 = time()
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    _sync_cuda_if_needed()
    epoch_profile["to_device"] += time() - t0

    _sync_cuda_if_needed()
    t0 = time()
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    _sync_cuda_if_needed()
    epoch_profile["shuffle"] += time() - t0

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    _sync_cuda_if_needed()
    loop_t0 = time()
    for (batch_i,
        (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        sample_weights = None
        if mix_state is not None:
            _sync_cuda_if_needed()
            t0 = time()
            sample_weights = mix_state["user_weights"][batch_users.cpu().numpy()]
            _sync_cuda_if_needed()
            epoch_profile["weight_lookup"] += time() - t0
        _sync_cuda_if_needed()
        t0 = time()
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, sample_weights=sample_weights)
        _sync_cuda_if_needed()
        epoch_profile["stage_one"] += time() - t0
        aver_loss += cri
        if world.tensorboard:
            _sync_cuda_if_needed()
            t0 = time()
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            _sync_cuda_if_needed()
            epoch_profile["tb_log"] += time() - t0
    _sync_cuda_if_needed()
    epoch_profile["batch_loop"] += time() - loop_t0
    _sync_cuda_if_needed()
    epoch_profile["epoch_total"] += time() - epoch_t0

    for key, value in epoch_profile.items():
        _BPR_PROFILE_TOTALS[key] += value
    _BPR_PROFILE_EPOCHS += 1

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    epoch_profile_str = _format_profile_seconds(epoch_profile)
    cum_profile_str = _format_profile_seconds(_BPR_PROFILE_TOTALS)
    return f"loss{aver_loss:.3f}-{time_info}-epoch_profile={epoch_profile_str}-cum_profile={cum_profile_str}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0, allowed_users=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.get_eval_dict(world.config['eval_split'])
    if not testDict:
        raise ValueError(
            f"No users found in eval split '{world.config['eval_split']}'. "
            "Check that the split file exists and is non-empty."
        )
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        if allowed_users is not None:
            users = [u for u in users if u in allowed_users]
            if not users:
                raise ValueError("No evaluation users remain after source-group filtering.")
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
