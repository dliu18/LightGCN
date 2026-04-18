'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-1m]")
    parser.add_argument('--eval_split', type=str, default='test',
                        help="evaluation split: [test, val]")
    parser.add_argument('--val_split_idx', type=int, default=0,
                        help="validation fold index used when eval_split=val")
    parser.add_argument('--group_mixing', type=int, default=0,
                        help="enable group-aware data mixing (0/1)")
    parser.add_argument('--group_labels_pkl', type=str, default='',
                        help="pickle file containing feature->label->user_indices")
    parser.add_argument('--feature_name', type=str, default='',
                        help="feature key in labels pickle, e.g. Age")
    parser.add_argument('--source_group', type=str, default='',
                        help="source group label within feature, e.g. 18")
    parser.add_argument('--alpha_aug', type=float, default=1.0,
                        help="augmentation scaling scalar")
    parser.add_argument('--alpha_mix', type=str, default='',
                        help=("augmentation group ratio weights, either JSON dict "
                              "or comma list in sorted augmentation-label order"))
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    return parser.parse_args()
