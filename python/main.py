import os
import time
import torch
import argparse

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)             # 数据集名称
parser.add_argument('--train_dir', required=True)           # 训练结果保存目录
parser.add_argument('--batch_size', default=128, type=int)  # 批次大小
parser.add_argument('--lr', default=0.001, type=float)      # 学习率
parser.add_argument('--maxlen', default=200, type=int)      # 用户行为序列的最大长度
parser.add_argument('--hidden_units', default=50, type=int) # 隐藏层大小
parser.add_argument('--num_blocks', default=2, type=int)    # Transformer层数
parser.add_argument('--num_epochs', default=1000, type=int) # 训练论数
parser.add_argument('--num_heads', default=1, type=int)     # 注意力头数
parser.add_argument('--dropout_rate', default=0.2, type=float) # Dropout率
parser.add_argument('--l2_emb', default=0.0, type=float)    # L2正则化
parser.add_argument('--device', default='cuda', type=str)   # 设备
parser.add_argument('--inference_only', default=False, type=str2bool) # 是否只进行推理
parser.add_argument('--state_dict_path', default=None, type=str)      # 预训练模型路径
parser.add_argument('--norm_first', action='store_true', default=False) # 是否Pre-norm

parser.add_argument('--time_range', default=25, type=int)    # 时间分桶桶数
parser.add_argument('--time_func', default='log', type=str)  # 时间映射函数
parser.add_argument('--time_scale', default=1.0, type=float) # 缩放因子

parser.add_argument('--short_num_blocks', default=2, type=int)    # 短期CNN层数
parser.add_argument('--short_kernel_size', default=3, type=int)   # 短期卷积核大小
parser.add_argument('--recent_window', default=5, type=int)       # 最近交互紧凑度窗口
parser.add_argument('--gate_hidden_units', default=64, type=int)  # 门控MLP隐藏层大小

parser.add_argument('--use_cnn', default=True, type=str2bool)     # 是否启用短期CNN分支与门控融合

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # 构建索引
    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset 加载数据集并分割为训练集、验证集、测试集
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    # 计算批次数
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    # 计算平均序列长度
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # 日志
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    # 初始化采样器和模型
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    # 初始化模型参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data) # Xavier正态分布初始化
        except:
            pass # just ignore those failed init layers

    # 初始化位置嵌入和物品嵌入的0位置
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.time_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1

    # 加载预训练模型（如果提供了路径）
    if args.state_dict_path is not None:
        try:
            load_result = model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)), strict=False)
            if len(load_result.missing_keys) > 0:
                print('missing keys while loading checkpoint:', load_result.missing_keys)
            if len(load_result.unexpected_keys) > 0:
                print('unexpected keys while loading checkpoint:', load_result.unexpected_keys)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    # 只评估（--inference_only）
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # 定义损失函数和优化器
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    # best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()

    patience = 5 # 早停参数
    patience_cnt = 0

    # 训练循环
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition

        epoch_loss = 0.0

        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # 获取批次数据
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            log_seqs = seq[:, 0, :]
            time_seqs = seq[:, 1, :]

            # 模型前向传播
            pos_logits, neg_logits = model(u, log_seqs, time_seqs, pos, neg) # 使用forward方法

            # 创建标签
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            
            # 计算损失
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0) # 只对非零位置计算损失
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            # L2正则化防过拟合
            # torch.norm(param) returns the square root of the sum of squared weights (‖w‖₂), 
            # should be torch.norm(param)**2 or the way below which is faster.
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.sum(param ** 2) 
            for param in model.time_emb.parameters(): loss += args.l2_emb * torch.sum(param ** 2)   
            for param in model.time_cont_proj.parameters(): loss += args.l2_emb * torch.sum(param ** 2)
            if model.gate_network is not None:
                for param in model.gate_network.parameters(): loss += args.l2_emb * torch.sum(param ** 2)
            if len(model.short_layers) > 0:
                for param in model.short_layers.parameters(): loss += args.l2_emb * torch.sum(param ** 2)
            
            # 反向传播
            loss.backward()
            adam_optimizer.step()

            epoch_loss += loss.item()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        print("loss in epoch {}: {}".format(epoch, epoch_loss / num_batch))

        # 验证集评估
        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            # 保存最佳
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                # best_test_ndcg = max(t_test[0], best_test_ndcg)
                # best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

                patience_cnt = 0
            else:
                patience_cnt += 1

            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()

            # 早停
            if patience_cnt >= patience:
                print('early stop')
                break

            t0 = time.time()
            model.train()
    
        # 保存最终
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
