# utils.py in SASRec
# Usage: 1) Dataset Construction 2) Evaluate
import sys
import copy
import torch
import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
# ----- Dataset Construction by Split Whole Dataset into TRAIN/VALIDATION/TEST Dataset -----

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def data_partition_with_sample(fname,samplek):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = []
    user_test = []
    # assume user/item index starting from 1
    f = open('%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
        else:
            user_train[user] = User[user][:-2]
            user_valid.append((user, User[user][-2]))
            user_test.append((user, User[user][-1]))

    # 随机选择10000条验证集数据
    if len(user_valid) > samplek:
        user_valid = random.sample(user_valid, samplek)
        
    # 随机选择10000条测试集数据
    if len(user_test) > samplek:
        user_test = random.sample(user_test, samplek)

    return [user_train, user_valid, user_test, usernum, itemnum]

# ----- Evaluation Function in SASRec -----

# # evaluate on val set
# def evaluate_valid(model, dataset, maxlen):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue

#         seq = np.zeros([maxlen], dtype=np.int32)
#         idx = maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break

#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, maxlen, device):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    print(len(train))
    NDCG200 = 0.0
    HT200 = 0.0
    NDCG50 = 0.0
    HT50 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    valid_user = 0.0
    maxK=2500
    MRR=0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(maxK-1):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        pred_data = [np.array(l) for l in [[u], [seq], item_idx]]
        tensor_data = [torch.tensor(nd) for nd in pred_data]
        pred_data = [td.to(device) for td in tensor_data]
        model.to(device)
        predictions = -model.predict(*pred_data)
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        
        if rank < maxK-2:
            MRR+=1/(rank+1)
            if rank < 200:
                NDCG200 += 1 / np.log2(rank + 2)
                HT200 += 1
                if rank < 50:
                    NDCG50 += 1 / np.log2(rank + 2)
                    HT50 += 1
                    if rank < 10:
                        NDCG10 += 1 / np.log2(rank + 2)
                        HT10 += 1

    return NDCG200 / valid_user,NDCG50 / valid_user,NDCG10 / valid_user, HT200 / valid_user, HT50 / valid_user, HT10 / valid_user,MRR/valid_user

def evaluate_with_log(model, dataset, maxlen, device):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    print(len(train))
    NDCG200 = 0.0
    HT200 = 0.0
    NDCG50 = 0.0
    HT50 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    valid_user = 0.0
    maxK=2500
    MRR=0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    tosave_train={}
    idx=0
    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        tosave_train[u]=[train[u],test[u]]
        idx+=1
        if idx>1000: break
    filename = 'data_watch.json'
    with open(filename, 'w') as file:
        json.dump(tosave_train, file, indent=4) 
    ranks=[]
    idx=0
    pred=[]
    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(maxK-1):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        pred_data = [np.array(l) for l in [[u], [seq], item_idx]]
        tensor_data = [torch.tensor(nd) for nd in pred_data]
        pred_data = [td.to(device) for td in tensor_data]
        model.to(device)
        predictions = -model.predict(*pred_data)
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()
        ranks.append(rank)
        valid_user += 1
        
        MRR+=1/(rank+1)
        if rank < 200:
            NDCG200 += 1 / np.log2(rank + 2)
            HT200 += 1
            if rank < 50:
                NDCG50 += 1 / np.log2(rank + 2)
                HT50 += 1
                if rank < 10:
                    NDCG10 += 1 / np.log2(rank + 2)
                    HT10 += 1
    if np.sum(ranks[0:50])==0:
        print(pred_data)
        input()
        print(predictions)
    print(valid_user)
    print(ranks[0:50])
    filename = 'rank_watch.json'
    with open(filename, 'w') as file:
        json.dump(ranks, file, indent=4) 
    return NDCG200 / valid_user,NDCG50 / valid_user,NDCG10 / valid_user, HT200 / valid_user, HT50 / valid_user, HT10 / valid_user,MRR/valid_user
