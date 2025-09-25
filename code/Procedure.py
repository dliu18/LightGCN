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
        cri = results["loss"]

        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            # w.add_scalar(f'BPRLoss/Item Pairs', num_item_pairs, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            # w.add_scalar(f'BPRLoss/Low Popularity Similarity', low_pop_similarity, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    quadrant_labels = X[2]

    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    quadrant_recalls = [[], [], [], []]

    for k in world.topks[:-1]:
        # all users
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])

        for quadrant_idx in range(len(quadrant_labels)):
            user_labels = quadrant_labels[quadrant_idx]
            if np.sum(user_labels) > 0:
                ret = utils.RecallPrecision_ATk(groundTrue[user_labels], r[user_labels], k)
                quadrant_recalls[quadrant_idx].append(ret['recall'])
            else:
                quadrant_recalls[quadrant_idx].append(0.0)  

        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre),
            'ndcg':np.array(ndcg),
            'low_low_recall': np.array(quadrant_recalls[0]),
            'low_high_recall': np.array(quadrant_recalls[1]),
            'high_low_recall': np.array(quadrant_recalls[2]),
            'high_high_recall': np.array(quadrant_recalls[3]),
            }
        
def popularity_opportunity_one_batch(X):
    sorted_items_batch = X[0].numpy()
    groundTrue_batch = X[1]
    quadrant_labels = X[2]

    # max_k = world.topks[-1]
    agg_item_freqs_and_ranks = {}
    sub_item_freqs_and_ranks = []
    for _ in range(len(quadrant_labels)):
        sub_item_freqs_and_ranks.append({})
    for user_idx in range(len(sorted_items_batch)):
        groundTrue = groundTrue_batch[user_idx]
        sorted_items = sorted_items_batch[user_idx]

        actual_subgroup_idx = -1
        for candidate_subgroup_idx, labels in enumerate(quadrant_labels):
            if labels[user_idx] > 0:
                actual_subgroup_idx = candidate_subgroup_idx
                break
        assert actual_subgroup_idx >= 0

        # use below when max_k is not truncated
        pred_ranks = np.array([
            np.where(sorted_items == item)[0][0] + 1 \
                for item in groundTrue
        ])

        ## Use below when max_k is truncated
        # pred_ranks = np.array([
        #     np.where(sorted_items == item)[0][0] + 1 \
        #         if item in sorted_items else max_k \
        #         for item in groundTrue
        # ])
        for item_idx, item in enumerate(groundTrue):
            if item not in agg_item_freqs_and_ranks:
                agg_item_freqs_and_ranks[item] = [0, 0]
            if item not in sub_item_freqs_and_ranks[actual_subgroup_idx]:
                sub_item_freqs_and_ranks[actual_subgroup_idx][item] = [0, 0]
            agg_item_freqs_and_ranks[item][0] += 1
            agg_item_freqs_and_ranks[item][1] += pred_ranks[item_idx]
            sub_item_freqs_and_ranks[actual_subgroup_idx][item][0] += 1
            sub_item_freqs_and_ranks[actual_subgroup_idx][item][1] += pred_ranks[item_idx]

    return [agg_item_freqs_and_ranks, sub_item_freqs_and_ranks]


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

def Test(dataset, Recmodel, epoch, w=None, multicore=0, is_test=True):
    u_batch_size = world.config['test_u_batch_size']
    label = "Test" if is_test else "Train"
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    trainDict: dict = dataset.trainDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()

    # max_K = max(world.topks)
    max_K = dataset.m_items
    
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
                'precision': np.zeros(len(world.topks) - 1),
                'recall': np.zeros(len(world.topks) - 1),
                'ndcg': np.zeros(len(world.topks) - 1),
                'low_low_recall': np.zeros(len(world.topks) - 1),
                'low_high_recall': np.zeros(len(world.topks) - 1),
                'high_low_recall': np.zeros(len(world.topks) - 1),
                'high_high_recall': np.zeros(len(world.topks) - 1) #this could be re-written for general groups
            }
    with torch.no_grad():
        users = list(testDict.keys())
        if not is_test:
            users = list(trainDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
       
        user_quadrant_labels = dataset.user_quadrant_labels
        rating_list = []
        groundTrue_list = []
        quadrant_labels_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = np.array([testDict[u] for u in batch_users], dtype=object)
            if not is_test:
                groundTrue = np.array([trainDict[u] for u in batch_users], dtype=object)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            if world.config["pc_alpha"] > 0:
                rating = utils.postprocess_rating(batch_users_gpu, rating, dataset)
            #rating = rating.cpu()

            if is_test:
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
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            quadrant_labels_list.append([user_quadrant_labels[quadrant_idx][batch_users] for quadrant_idx in range(len(user_quadrant_labels))])
        X = zip(rating_list, groundTrue_list, quadrant_labels_list)
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
            results['low_low_recall'] += result['low_low_recall']
            results['low_high_recall'] += result['low_high_recall']
            results['high_low_recall'] += result['high_low_recall']
            results['high_high_recall'] += result['high_high_recall']

        # debug the quadrant recalls 
        # print(f"total recall: {results['recall']}")
        # print(f"total quadrant recall: {results['low_low_recall'] + results['low_high_recall'] + results['high_low_recall'] +results['high_high_recall']}")
        # print(f"total low low recall: {results['low_low_recall']}")
        # print(f"total low high recall: {results['low_high_recall']}")
        # print(f"total high low recall: {results['high_low_recall']}")
        # print(f"total high high recall: {results['high_high_recall']}")
        # print(f"total quadrant users: {np.sum([np.sum(labels) for labels in user_quadrant_labels])}")

        num_niche_users = np.sum(dataset.is_niche_user(users))
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))       
        results['ndcg'] /= float(len(users))

        # truncate the quadrant labels for the data shapley setting. When training on the entire training set,
        # the truncation is a no-op.
        results['low_low_recall'] /= float(np.sum(user_quadrant_labels[0][:len(users)])) 
        results['low_high_recall'] /= float(np.sum(user_quadrant_labels[1][:len(users)]))
        results['high_low_recall'] /= float(np.sum(user_quadrant_labels[2][:len(users)]))
        results['high_high_recall'] /= float(np.sum(user_quadrant_labels[3][:len(users)]))
        # results['auc'] = np.mean(auc_record)

        item_freq_in_predictions = {item: 0 for item in range(dataset.m_items)}
        for item_freq_in_predictions_batch in gini_coef_batches:
            for item in item_freq_in_predictions_batch:
                item_freq_in_predictions[item] += item_freq_in_predictions_batch[item]
        item_ratios = np.array([item_freq_in_predictions[item] / (dataset.n_users * world.topks[0]) for item in range(dataset.m_items)])
        results["gini-index"] = utils.gini_index(dataset.item_popularities, item_ratios)

        def _extract_avg_ranks(popularity_opportunity_batches):
            avg_ranks = []
            label_names = ["agg", "low_low", "low_high", "high_low", "high_high"]

            ## Agg
            item_freqs_and_ranks = {item: [0, 0] for item in range(dataset.m_items)}
            for popularity_opportunity_batch in popularity_opportunity_batches:
                item_freqs_and_ranks_batch = popularity_opportunity_batch[0]
                for item in item_freqs_and_ranks_batch:
                    item_freqs_and_ranks[item][0] += item_freqs_and_ranks_batch[item][0]
                    item_freqs_and_ranks[item][1] += item_freqs_and_ranks_batch[item][1]
            # compute avg rank 
            agg_avg_ranks = np.array([item_freqs_and_ranks[item][1] / item_freqs_and_ranks[item][0] \
                if item_freqs_and_ranks[item][0] > 0 else -1 for item in range(dataset.m_items)])
            avg_ranks.append(agg_avg_ranks)

            for subgroup_idx in range(len(label_names) - 1):
                item_freqs_and_ranks = {item: [0, 0] for item in range(dataset.m_items)}
                for popularity_opportunity_batch in popularity_opportunity_batches:
                    item_freqs_and_ranks_batch = popularity_opportunity_batch[1][subgroup_idx]
                    for item in item_freqs_and_ranks_batch:
                        item_freqs_and_ranks[item][0] += item_freqs_and_ranks_batch[item][0]
                        item_freqs_and_ranks[item][1] += item_freqs_and_ranks_batch[item][1]
                # compute avg rank 
                subgroup_avg_ranks = np.array([item_freqs_and_ranks[item][1] / item_freqs_and_ranks[item][0] \
                    if item_freqs_and_ranks[item][0] > 0 else -1 for item in range(dataset.m_items)])
                avg_ranks.append(subgroup_avg_ranks)

            ## Subgroups
            return label_names, avg_ranks

        label_names, extracted_avg_ranks = _extract_avg_ranks(popularity_opportunity_batches)
        for idx in range(len(label_names)):
            avg_ranks = extracted_avg_ranks[idx]
            results[f"{label_names[idx]}_popularity-opportunity-bias"] = utils.pop_opp_bias(
                dataset.item_popularities[avg_ranks > 0],
                avg_ranks[avg_ranks > 0])

            if is_test and world.tensorboard:
                w.add_scalar(
                    f'Quadrants/{label_names[idx]}_Popularity Opportunity Bias@{world.topks[0]}',
                    results[f"{label_names[idx]}_popularity-opportunity-bias"],
                    epoch
                )

        if world.tensorboard:
            w.add_scalars(f'{label}/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'{label}/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks) - 1)}, epoch)
            w.add_scalars(f'{label}/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks) - 1)}, epoch)
            
            
            # recall by quadrant
            if is_test:
                w.add_scalars(f'Quadrants/Low_Low_Recall@{world.topks}',
                              {str(world.topks[i]): results['low_low_recall'][i] for i in range(len(world.topks) - 1)}, epoch)
                w.add_scalars(f'Quadrants/Low_High_Recall@{world.topks}',
                              {str(world.topks[i]): results['low_high_recall'][i] for i in range(len(world.topks) - 1)}, epoch)
                w.add_scalars(f'Quadrants/High_Low_Recall@{world.topks}',
                              {str(world.topks[i]): results['high_low_recall'][i] for i in range(len(world.topks) - 1)}, epoch)
                w.add_scalars(f'Quadrants/High_High_Recall@{world.topks}',
                              {str(world.topks[i]): results['high_high_recall'][i] for i in range(len(world.topks) - 1)}, epoch)
               
            # popularity-bias metrics
            w.add_scalar(
                f'{label}/Gini@{world.topks[0]}',
                results["gini-index"],
                epoch
            )

            w.add_scalar(
                f'{label}/Popularity Opportunity Bias@{world.topks[0]}',
                results["agg_popularity-opportunity-bias"],
                epoch
            )

        if multicore == 1:
            pool.close()
        print(results)
        return results
