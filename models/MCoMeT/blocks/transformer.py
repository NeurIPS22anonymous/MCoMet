import torch.nn as nn
from utils import split_first_dim_linear
import math
from itertools import combinations 
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.models as models
from nncore.nn import (MODELS, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


@MODELS.register()
class BottleneckTransformerLayer(nn.Module):

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BottleneckTransformerLayer, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att3 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att4 = MultiHeadAttention(dims, heads=heads, p=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)
        self.ffn2 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)
        self.norm5 = build_norm_layer(norm_cfg, dims=dims)
        self.norm6 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, a, b, t, pe=None, mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)



        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        at = self.att1(dt, ka, da, mask=mask)
        bt = self.att2(dt, kb, db, mask=mask)

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        a = a + self.att3(qa, dt)
        b = b + self.att4(qb, dt)

        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t


@MODELS.register()
class BottleneckTransformer(nn.Module):
    def __init__(self, dims, num_tokens=4, temporal_set_size=3, num_layers=1, dropout=0.5, seq_len=256, **kwargs):
        super(BottleneckTransformer, self).__init__()

        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        max_len =  int(seq_len*1.5)
        self.pe = PositionalEncoding(dims,dropout, max_len=max_len)
        self.q_linear = nn.Linear(dims* temporal_set_size, dims)#.cuda()
        self.k_linear = nn.Linear(dims * temporal_set_size, dims)#.cuda()
        self.norm_q = nn.LayerNorm(dims)
        self.norm_k = nn.LayerNorm(dims)
        # implementation of temporal_relation_module
        frame_idxs = [i for i in range(seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
     
        self.token = Parameter(num_tokens, dims)
        self.encoder = nn.ModuleList([
            BottleneckTransformerLayer(dims, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, a, b, **kwargs):            
        n_queries = a.size(0)
        n_keys = b.size(0)
        q = [torch.index_select(a, -2, p).reshape(n_queries, -1) for p in self.tuples]
        queries = torch.stack(q, dim=-2)
        k = [torch.index_select(b, -2, p).reshape(n_keys, -1) for p in self.tuples]
        keys = torch.stack(k,dim=-2)
        a = self.norm_q(self.q_linear(querys))
        b = self.norm_k(self.k_linear(keys))
        t = self.token.expand(a.size(0), -1, -1)
        for enc in self.encoder:
            a, b, t = enc(a, b, t, **kwargs)
        return a, b