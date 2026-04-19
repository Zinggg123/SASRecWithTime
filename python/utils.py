import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):
    """
    构建用户-物品索引映射
    读取数据文件，构建用户到物品列表的映射和物品到用户列表的映射
    """
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    """
    在范围[l,r)内随机选择一个不在集合s中的数字
    用于负采样（negative sampling）
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    """
    数据采样函数，用于生成训练批次
    """
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        item_seq = np.zeros([maxlen], dtype=np.int32)  # 输入序列的物品
        time_seq = np.zeros([maxlen], dtype=np.int32)  # 输入序列的时间戳
        pos = np.zeros([maxlen], dtype=np.int32)  # 正样本（下一个要预测的物品）
        neg = np.zeros([maxlen], dtype=np.int32)  # 负样本（随机采样的未交互物品）

        ts = set([x[0] for x in user_train[uid]])

        nxt = user_train[uid][-1][0]
        idx = maxlen - 1
        
        train_seq = user_train[uid]

        for i, _ in reversed(user_train[uid][:-1]):
            item_seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for k in range(len(train_seq) - 1, -1, -1):
            _, t = train_seq[k]
            t_prev = train_seq[k - 1][1] if k > 0 else t
            inte = max(0, t - t_prev) 
            time_seq[idx] = inte
            idx -= 1
            if idx == -1: break

        # idx = maxlen - 1
        # #for i, t in reversed(user_train[uid][:-1]):
        # for k in range(len(train_seq) - 1, -1, -1):
        #     i, t = train_seq[k]
        #     # 计算与上一次交互的时间间隔。如果是第一个交互，间隔记为 0
        #     t_prev = train_seq[k - 1][1] if k > 0 else t
        #     interval = max(0, t - t_prev) 

        #     item_seq[idx] = i
        #     time_seq[idx] = interval
        #     pos[idx] = nxt
        #     neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
        #     nxt = i
        #     idx -= 1
        #     if idx == -1: break

        return (uid, [item_seq, time_seq], pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True: # 守护线程，一直运行到主程序结束
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """
    并行数据采样器
    """
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True # 设置为守护线程
            self.processors[-1].start() # 启动线程

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    """
    数据分割函数：将数据分为训练集、验证集、测试集
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = int(float(t)) # 时间戳
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i,t))

    # 分割数据：最后两个交互分别作为验证和测试
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
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

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    """
    在测试集上评估模型性能
    计算NDCG@10和HR@10指标
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        # 构建测试的输入序列（加上验证集物品）
        item_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        item_seq[idx] = valid[u][0][0]
        idx -= 1
        for i,_ in reversed(train[u]):
            item_seq[idx] = i
            idx -= 1
            if idx == -1: break

        full_seq = train[u] + [valid[u][0]]

        idx = args.maxlen - 1
        time_seq[idx] = max(0, test[u][0][1] - full_seq[-1][1])
        idx -= 1
        for k in range(len(full_seq) - 1, -1, -1):
            _, t = full_seq[k]
            t_prev = full_seq[k - 1][1] if k > 0 else t
            inte = max(0, t - t_prev) 
            time_seq[idx] = inte
            idx -= 1
            if idx == -1: break


        # # item_seq[idx] = valid[u][0][0]
        # # time_seq[idx] = valid[u][0][1]
        # # idx -= 1
        # # for i, t in reversed(train[u]):
        # for k in range(len(full_seq) - 1, -1, -1):
        #     i, t = full_seq[k]
        #     t_prev = full_seq[k - 1][1] if k > 0 else t
        #     interval = max(1, t - t_prev) #防止log出错

        #     item_seq[idx] = i
        #     time_seq[idx] = interval
        #     idx -= 1
        #     if idx == -1: break

        # 构造1+100候选物品列表
        rated = set([x[0] for x in train[u]])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [item_seq], [time_seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        # 计算排名
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    """
    在验证集上评估模型性能
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        # 构建输入序列（只包含训练集物品）
        item_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        for i,_ in reversed(train[u]):
            item_seq[idx] = i
            idx -= 1
            if idx == -1: break

        full_seq = train[u]

        idx = args.maxlen - 1
        time_seq[idx] = max(0, valid[u][0][1] - full_seq[-1][1])
        idx -= 1
        for k in range(len(full_seq) - 1, -1, -1):
            _, t = full_seq[k]
            t_prev = full_seq[k - 1][1] if k > 0 else t
            inte = max(0, t - t_prev) 
            time_seq[idx] = inte
            idx -= 1
            if idx == -1: break
        
        # # for i, t in reversed(train[u]):
        # for k in range(len(full_seq) - 1, -1, -1):
        #     i, t = full_seq[k]
        #     t_prev = full_seq[k - 1][1] if k > 0 else t
        #     interval = max(0, t - t_prev)
        
        #     item_seq[idx] = i
        #     time_seq[idx] = interval
        #     idx -= 1
        #     if idx == -1: break

        # 候选物品列表1+100
        rated = set([x[0] for x in train[u]])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [item_seq], [time_seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
