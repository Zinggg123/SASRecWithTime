import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    """
    位置点前馈网络层
    用于SASRec模型中的前馈层
    """
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        # 定义网络层
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # 一维卷积（等价于全连接）
        self.dropout1 = torch.nn.Dropout(p=dropout_rate) # Dropout层
        self.relu = torch.nn.ReLU() # ReLU激活函数
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # 一维卷积（等价于全连接）
        self.dropout2 = torch.nn.Dropout(p=dropout_rate) # Dropout层

    def forward(self, inputs):
        # 前向传播：定义数据如何通过网络（input做转置以适应Conv1d）
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    """
    SASRec (Self-Attentive Sequential Recommendation) 模型
    基于自注意力机制的序列推荐模型
    """
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.time_num = args.time_range
        self.time_scale = args.time_scale
        self.time_func = args.time_func

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        
        # 嵌入层：物品嵌入和位置嵌入（离散ID -> 连续的向量表示）
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.time_emb = torch.nn.Embedding(self.time_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 多层Transformer结构
        self.attention_layernorms = torch.nn.ModuleList() # 每层的LayerNorm层 # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()     # 每层的Multi-head Attention层
        self.forward_layernorms = torch.nn.ModuleList()   # 每层的前馈网络的LayerNorm
        self.forward_layers = torch.nn.ModuleList()       # 每层的前馈网络

        # 最后的LayerNorm
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 构建多层Transformer块
        for _ in range(args.num_blocks):
            # 注意力层的LayerNorm
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # 多头注意力层
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            # 前馈层的LayerNorm
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 前馈网络层
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, time_seqs): # TODO: fp64 and int64 as default in python, trim?
        """
        将用户行为序列转换为 结果特征表示
        """

        # 物品嵌入
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5 # /√d 缩放嵌入向量

        # 位置嵌入
        # 创建位置索引矩阵
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)  # 掩码，非0位置才添加位置信息
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))

        # 时间嵌入
        if self.time_func == 'log':
            times = self.time_scale * np.log(time_seqs + 1)
        else:
            times = self.time_num * (1 - np.exp(-time_seqs * self.time_scale))

        times = times.astype(np.int64)
        times = np.clip(times, 1, self.time_num)
        times *= (log_seqs != 0)
        seqs += self.time_emb(torch.LongTensor(times).to(self.dev))

        seqs = self.emb_dropout(seqs) # dropout

        # 创建因果掩码，确保只能看到当前位置及之前的序列
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # 通过多层Transformer块
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                # Pre-norm: 先归一化再进行注意力计算
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs  # 残差连接
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # 先注意力计算再归一化
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # 最终的特征表示 # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs): # for training        
        """
        前向传播，用于训练
        """
        # log_seqs = seq[0]
        # time_seqs = seq[1]

        # 获取序列特征表示
        log_feats = self.log2feats(log_seqs, time_seqs) # user_ids hasn't been used yet

        # 获取正样本、负样本的嵌入
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # 计算正样本、负样本的得分
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_seqs, item_indices): # for inference
        """
        预测函数，用于推理？
        """
        # log_seqs = seq[0]
        # time_seqs = seq[1]

        log_feats = self.log2feats(log_seqs, time_seqs) # user_ids hasn't been used yet

        # 只使用序列的最后一个位置的特征
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        # 获取候选物品的嵌入
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        # 计算用户特征与候选物品相似度
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
