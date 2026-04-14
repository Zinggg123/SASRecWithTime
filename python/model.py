import math

import numpy as np
import torch
import torch.nn.functional as F


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


class CausalConvBlock(torch.nn.Module):
    """
    因果卷积块
    用于短期分支，保证卷积只利用当前位置及其历史信息
    """
    def __init__(self, hidden_units, dropout_rate, kernel_size):
        super(CausalConvBlock, self).__init__()

        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(hidden_units,
                                    hidden_units,
                                    kernel_size=kernel_size,
                                    padding=0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.activation = torch.nn.GELU()
        self.layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, inputs):
        residual = inputs
        outputs = inputs.transpose(-1, -2)
        outputs = F.pad(outputs, (self.kernel_size - 1, 0))
        outputs = self.conv(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.transpose(-1, -2)
        return self.layernorm(residual + outputs)

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

        self.recent_window = getattr(args, 'recent_window', 5)
        self.short_kernel_size = getattr(args, 'short_kernel_size', 3)
        self.short_num_blocks = getattr(args, 'short_num_blocks', 2)
        self.gate_hidden_units = getattr(args, 'gate_hidden_units', args.hidden_units)
        self.time_norm = math.log1p(max(self.time_num, 1))

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        
        # 嵌入层：物品嵌入和位置嵌入（离散ID -> 连续的向量表示）
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.time_emb = torch.nn.Embedding(self.time_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_cont_proj = torch.nn.Linear(3, args.hidden_units, bias=False)

        # 多层Transformer结构
        self.attention_layernorms = torch.nn.ModuleList() # 每层的LayerNorm层 # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()     # 每层的Multi-head Attention层
        self.forward_layernorms = torch.nn.ModuleList()   # 每层的前馈网络的LayerNorm
        self.forward_layers = torch.nn.ModuleList()       # 每层的前馈网络

        # 短期并行分支：用因果CNN捕获局部冲动模式
        self.short_layers = torch.nn.ModuleList()
        for _ in range(self.short_num_blocks):
            self.short_layers.append(CausalConvBlock(args.hidden_units, args.dropout_rate, self.short_kernel_size))
        self.short_output_norm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 动态门控：由时间间隔、最近交互紧凑度和两路表示共同决定融合比例
        gate_input_dim = args.hidden_units * 3 + 2
        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(gate_input_dim, self.gate_hidden_units),
            torch.nn.GELU(),
            torch.nn.Dropout(p=args.dropout_rate),
            torch.nn.Linear(self.gate_hidden_units, 1)
        )

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

    def _recent_compactness(self, time_values, valid_mask):
        """
        基于滑动窗口计算最近交互的时间紧凑度。
        值越大表示近期交互越密集，更偏向短期兴趣。
        """
        window = max(1, min(self.recent_window, time_values.size(1)))
        kernel = torch.ones((1, 1, window), device=time_values.device, dtype=time_values.dtype)

        summed_values = F.conv1d((time_values * valid_mask).unsqueeze(1), kernel, padding=window - 1)
        summed_mask = F.conv1d(valid_mask.unsqueeze(1), kernel, padding=window - 1)

        summed_values = summed_values.squeeze(1)[:, :time_values.size(1)]
        summed_mask = summed_mask.squeeze(1)[:, :time_values.size(1)]

        mean_gap = summed_values / (summed_mask + 1e-8)
        compactness = torch.exp(-mean_gap)
        return compactness * valid_mask

    def _build_time_features(self, log_seqs, time_seqs):
        """
        同时构建离散时间桶、连续时间间隔和最近紧凑度特征。
        """
        log_tensor = torch.LongTensor(log_seqs).to(self.dev)
        time_tensor = torch.LongTensor(time_seqs).to(self.dev).float()
        valid_mask = (log_tensor != 0).float()

        if self.time_func == 'log':
            time_bucket = self.time_scale * torch.log1p(time_tensor)
        else:
            time_bucket = self.time_num * (1 - torch.exp(-time_tensor * self.time_scale))

        time_bucket = time_bucket.long().clamp(1, self.time_num)
        time_bucket = time_bucket * valid_mask.long()
        time_bucket_emb = self.time_emb(time_bucket)

        normalized_gap = torch.log1p(time_tensor) / self.time_norm
        normalized_gap = normalized_gap * valid_mask

        recent_compactness = self._recent_compactness(time_tensor, valid_mask)
        recency_score = torch.exp(-normalized_gap) * valid_mask

        continuous_time = torch.stack([normalized_gap, recent_compactness, recency_score], dim=-1)
        time_continuous_emb = self.time_cont_proj(continuous_time)

        time_context = time_bucket_emb + time_continuous_emb
        return time_context, recent_compactness.unsqueeze(-1), recency_score.unsqueeze(-1)

    def _encode_long_branch(self, seqs):
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        return seqs

    def _encode_short_branch(self, seqs):
        short_seqs = seqs
        for block in self.short_layers:
            short_seqs = block(short_seqs)
        return self.short_output_norm(short_seqs)

    def log2feats(self, log_seqs, time_seqs): # TODO: fp64 and int64 as default in python, trim?
        """
        将用户行为序列转换为融合后的长短期特征表示
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

        # 时间特征：离散时间桶 + 连续间隔 + 最近紧凑度
        time_context, recent_compactness, recency_score = self._build_time_features(log_seqs, time_seqs)
        seqs = seqs + time_context

        seqs = self.emb_dropout(seqs) # dropout

        # 长短期并行编码
        long_feats = self._encode_long_branch(seqs)
        short_feats = self._encode_short_branch(seqs)

        # 时间间隔驱动的动态门控：近期越紧凑，越偏向短期分支
        gate_inputs = torch.cat([long_feats, short_feats, time_context, recent_compactness, recency_score], dim=-1)
        gate = torch.sigmoid(self.gate_network(gate_inputs))

        log_feats = gate * short_feats + (1 - gate) * long_feats
        log_feats = self.last_layernorm(log_feats) # 最终的特征表示 # (U, T, C) -> (U, -1, C)
        log_feats = log_feats * torch.LongTensor(log_seqs).to(self.dev).ne(0).unsqueeze(-1).float()

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
