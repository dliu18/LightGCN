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
    if world.config["shuffle_users"]:
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
        results = bpr.stageOne(batch_users, batch_pos, batch_neg)
        cri, num_item_pairs, low_pop_similarity = results["loss"], results["num item pairs"], results["avg low pop similarity"]

        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'BPRLoss/Item Pairs', num_item_pairs, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'BPRLoss/Low Popularity Similarity', low_pop_similarity, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    is_niche_users = X[2]

    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, pre_niche, recall_niche = [], [], [], [], []
    for k in world.topks[:-1]:
        # all users
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])

        if np.sum(is_niche_users) > 0:
            ret = utils.RecallPrecision_ATk(groundTrue[is_niche_users], r[is_niche_users], k)
            pre_niche.append(ret['precision'])
            recall_niche.append(ret['recall'])
        else:
            pre_niche.append(0)
            recall_niche.append(0)

        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre),
            'niche recall': np.array(recall_niche),
            'niche precision': np.array(pre_niche), 
            'ndcg':np.array(ndcg)}
        
def popularity_opportunity_one_batch(X):
    sorted_items_batch = X[0].numpy()
    groundTrue_batch = X[1]

    max_k = world.topks[-1]
    item_freqs_and_ranks = {}
    for idx in range(len(sorted_items_batch)):
        groundTrue = groundTrue_batch[idx]
        sorted_items = sorted_items_batch[idx]

        pred_ranks = np.array([
            np.where(sorted_items == item)[0][0] + 1 \
                if item in sorted_items else max_k \
                for item in groundTrue
        ])

        for idx, item in enumerate(groundTrue):
            if item not in item_freqs_and_ranks:
                item_freqs_and_ranks[item] = [0, 0]
            item_freqs_and_ranks[item][0] += 1
            item_freqs_and_ranks[item][1] += pred_ranks[idx]
    return item_freqs_and_ranks


def gini_coef_one_batch(X):
    k = world.topks[0]
    item_freq_in_predictions = {}

    sorted_items_batch = X[0].numpy()
    for sorted_items in sorted_items_batch:
        items_in_prediction = sorted_items[:k]
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
    results = {
                'precision': np.zeros(len(world.topks) - 1),
                'recall': np.zeros(len(world.topks) - 1),
                'niche precision': np.zeros(len(world.topks) - 1),
                'niche recall': np.zeros(len(world.topks) - 1),
                'ndcg': np.zeros(len(world.topks) - 1)
            }
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        is_niche_users = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = np.array([testDict[u] for u in batch_users], dtype=object)
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
            is_niche_users.append(dataset.is_niche_user(batch_users))
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(is_niche_users)
        X = zip(rating_list, groundTrue_list, is_niche_users)
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
            results['niche recall'] += result['niche recall']
            results['niche precision'] += result['niche precision']
            results['ndcg'] += result['ndcg']
        num_niche_users = np.sum(dataset.is_niche_user(users))
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['niche recall'] /= float(num_niche_users)
        results['niche precision'] /= float(num_niche_users)        
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        item_freqs_and_ranks = {item: [0, 0] for item in range(dataset.m_items)}
        for item_freqs_and_ranks_batch in popularity_opportunity_batches:
            for item in item_freqs_and_ranks_batch:
                item_freqs_and_ranks[item][0] += item_freqs_and_ranks_batch[item][0]
                item_freqs_and_ranks[item][1] += item_freqs_and_ranks_batch[item][1]
        # compute avg rank 
        avg_ranks = np.array([item_freqs_and_ranks[item][1] / item_freqs_and_ranks[item][0] \
            if item_freqs_and_ranks[item][0] > 0 else -1 for item in range(dataset.m_items)])

        item_freq_in_predictions = {item: 0 for item in range(dataset.m_items)}
        for item_freq_in_predictions_batch in gini_coef_batches:
            for item in item_freq_in_predictions_batch:
                item_freq_in_predictions[item] += item_freq_in_predictions_batch[item]
        item_ratios = np.array([item_freq_in_predictions[item] / (dataset.n_users * world.topks[0]) for item in range(dataset.m_items)])
        
        # gini coefficient 

        # popularity-opportunity bias 
        results["gini-index"] = utils.gini_index(dataset.item_popularities, item_ratios)
        results["popularity-opportunity-bias"] = utils.pop_opp_bias(dataset.item_popularities, avg_ranks)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'Test/Niche Recall@{world.topks}',
                          {str(world.topks[i]): results['niche recall'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'Test/Niche Precision@{world.topks}',
                          {str(world.topks[i]): results['niche precision'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks) - 1)}, epoch)
            
            # popularity-bias metrics
            w.add_scalar(
                f'Test/Gini@{world.topks[0]}',
                results["gini-index"],
                epoch
            )

            w.add_scalar(
                f'Test/Popularity Opportunity Bias@{world.topks[0]}',
                results["popularity-opportunity-bias"],
                epoch
            )

        if multicore == 1:
            pool.close()
        print(results)
        return results
