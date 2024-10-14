# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:44:00 2023

@author: Administrator
"""
import torch
import torch.utils.data as Data
import torch.nn as nn
import math
import torch.nn.functional as F


class MyDataSet(Data.Dataset):
    def __init__(self, mz,intensity,weights):
        self.mz = mz
        self.intensity = intensity
        self.weights = weights
        
    def __len__(self):
        return len(self.mz)
    
    def __getitem__(self, idx):
        return self.mz[idx],self.intensity[idx],self.weights[idx]
    
class PeakEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class Embedding(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(Embedding, self).__init__()
        self.PeakEmbedding = PeakEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        x = self.PeakEmbedding(x)
        return x

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))      
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden,dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

def get_attn_pad_mask(x):
    atten_mask1 = (x == 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    atten_mask2 = atten_mask1.transpose(-2, -1)
    atten_mask = torch.add(atten_mask1,atten_mask2)
    return atten_mask

class MWFormer(nn.Module):
    def __init__(self, vocab_size,hidden, n_layers, attn_heads, dropout):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.embedding = Embedding(vocab_size,hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden, 1)
        self.activ2 = nn.ReLU()
        self.linear = nn.Linear(hidden, hidden)
        self.intensity_linear = nn.Linear(1, hidden)
        self.flatten = nn.Flatten()
    
    def get_mid_memory(self,mz_,intensity_):
        atten_mask = get_attn_pad_mask(mz_)
        inten = self.intensity_linear(intensity_)
        output = self.embedding(mz_)
        output = output+inten
        memory_mid = []
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, atten_mask)
            memory_mid.append(output)
        return memory_mid
    
    def forward(self, mz_,intensity_):
        atten_mask = get_attn_pad_mask(mz_)
        # atten_mask = (mz_ == 0).unsqueeze(1).repeat(1, mz_.size(1), 1).unsqueeze(1)
        inten = self.intensity_linear(intensity_)
        output = self.embedding(mz_)
        output = output+inten
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, atten_mask)
        out = output.sum(1)
        out = self.fc_out(out)
        return out













