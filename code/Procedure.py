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
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
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
        
def popularity_opportunity_one_batch(X):
    max_k = world.topks[-1]
    item_freqs_and_ranks = {}
    for sorted_items, groundTrue in X:
        groundTrue_np = groundTrue.numpy()
        sorted_items_np = sorted_items.numpy()

        pred_ranks = np.array([
            np.where(sorted_items_np == item)[0][0] + 1 \
                if item in sorted_items_np else max_k \
                for item in groundTrue_np
        ])

        for idx, item in enumerate(groundTrue_np):
            if item not in item_freqs_and_ranks:
                item_freqs_and_ranks[item] = [0, 0]
            item_freqs_and_ranks[item][0] += 1
            item_freq_in_predictions[item][1] += pred_ranks[idx]
    return item_freqs_and_ranks


def gini_coef_one_batch(X):
    k = world.topks[0]
    item_freq_in_predictions = {}
    for sorted_items, _ in X:
        items_in_prediction = sorted_items.numpy()[:k]
        for item in items_in_prediction:
            if item not in item_freq_in_predictions:
                item_freq_in_predictions[item] = 0
            item_freq_in_predictions[item] += 1
    return item_freq_in_predictions

def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
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
            gini_coef_batches = pool.map(gini_coef_one_batch, X)
            popularity_opportunity_batches = pool.map(popularity_opportunity_one_batch, X)
        else:
            pre_results, gini_coef_batches, popularity_opportunity_batches = [], [], []
            for x in X:
                pre_results.append(test_one_batch(x))
                gini_coef_batches.append(gini_coef_one_batch(x))
                popularity_opportunity_batches.append(popularity_opportunity_one_batch(x))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        item_freqs_and_ranks = {item: [0, 0] for item in range(dataset.m_items)}
        for item_freqs_and_ranks_batch in popularity_opportunity_batches:
            for item in item_freqs_and_ranks_batch:
                item_freqs_and_ranks[item][0] += item_freqs_and_ranks_batch[item][0]
                item_freqs_and_ranks[item][1] += item_freqs_and_ranks_batch[item][1]
        # compute avg rank 
        avg_ranks = [item_freqs_and_ranks[item][1] / item_freqs_and_ranks[item][0] \
            if item_freqs_and_ranks[item][0] > 0 else -1 for item in range(dataset.m_items)]

        item_freq_in_predictions = {item: 0 for item in range(dataset.m_items)}
        for item_freq_in_predictions_batch in gini_coef_batches:
            for item in item_freq_in_predictions_batch:
                item_freq_in_predictions[item] += item_freq_in_predictions_batch[item]
        item_ratios = [item_freq_in_predictions[item] / (dataset.n_users * world.topks[0]) for item in range(dataset.m_items)]
        
        # gini coefficient 

        # popularity-opportunity bias 

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            
            # popularity-bias metrics
            w.add_scalar(
                f'Test/Gini@{world.topks[0]}',
                utils.gini_index(dataset.item_popularities, item_ratios)
                epoch
            )

            w.add_scalar(
                f'Test/Popularity Opportunity Bias@{world.topks[0]}',
                utils.pop_opp_bias(dataset.item_popularities, avg_ranks),
                epoch
            )
            
        if multicore == 1:
            pool.close()
        print(results)
        return results
