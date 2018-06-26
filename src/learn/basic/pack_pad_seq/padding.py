# -*- coding: utf-8 -*-
"""
处理batch中句子长度不同的问题
reference:
[1] pytorch里的pack_padded_sequence和pad_packed_sequence解析
(https://blog.csdn.net/lssc4205/article/details/79474735)
"""
import numpy as np
import wordfreq

# 1. read sentence and convert to padding matrix
vocab = {}
token_id = 1
lengths = []

with open('test.txt', 'r') as f:
    for l in f:
        tokens = wordfreq.tokenize(l.strip(), 'en')
        lengths.append(len(tokens))
        for t in tokens:
            if t not in vocab:
                vocab[t] = token_id
                token_id += 1

x = np.zeros((len(lengths), max(lengths)))
l_no = 0
with open('test.txt', 'r') as f:
    for l in f:
        tokens = wordfreq.tokenize(l.strip(), 'en')
        for i in range(len(tokens)):
            x[l_no, i] = vocab[tokens[i]]
        l_no += 1
print(x)
print("-"*100)

# 2.使用pack_padded_sequence(Packs a Tensor containing padded sequences of variable length)
import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.Tensor(x))  # x[batch, seq]
lengths = torch.Tensor(lengths)   # lengths [batch]
_, idx_sort = torch.sort(lengths, dim=0, descending=True)
_, idx_unsort = torch.sort(idx_sort, dim=0)

x = x.index_select(0, idx_sort)
print(x.data.size())   # [batch, seq]
lengths = list(lengths[idx_sort])
print(lengths)
x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
print(x_packed)
print(x_packed.data.size())  # [83]
"""
PackedSequence(data=tensor([ 36.,  23.,   1.,  10.,  37.,  24.,   2.,  11.,  38.,  25.,
          3.,  12.,   2.,  26.,   4.,  13.,  39.,   4.,   5.,  14.,
         40.,  27.,   6.,  15.,   9.,  28.,   7.,  16.,  41.,  29.,
          8.,  17.,  38.,   4.,   9.,  18.,   2.,  23.,   1.,   2.,
         42.,  30.,   7.,   7.,  40.,  19.,   8.,  19.,   2.,  14.,
          4.,  14.,  43.,  31.,   5.,  20.,  44.,  32.,   6.,  21.,
         14.,  33.,   2.,   4.,   1.,  34.,   3.,  22.,  45.,  16.,
          2.,  35.,  43.,  46.,  47.,   2.,  48.,  49.,  50.,   2.,
         43.,  51.,  52.]), batch_sizes=tensor([ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1]))
"""
# unpack (还原成原来的tensor)
x_padded = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
output = x_padded[0].index_select(0, idx_unsort)
print(output)
"""
tensor([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,   1.,
           7.,   8.,   4.,   5.,   6.,   2.,   3.,   0.,   0.,   0.,
           0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
        [ 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,   2.,
           7.,  19.,  14.,  20.,  21.,   4.,  22.,   0.,   0.,   0.,
           0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
        [ 23.,  24.,  25.,  26.,   4.,  27.,  28.,  29.,   4.,  23.,
          30.,  19.,  14.,  31.,  32.,  33.,  34.,  16.,  35.,   0.,
           0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
        [ 36.,  37.,  38.,   2.,  39.,  40.,   9.,  41.,  38.,   2.,
          42.,  40.,   2.,  43.,  44.,  14.,   1.,  45.,   2.,  43.,
          46.,  47.,   2.,  48.,  49.,  50.,   2.,  43.,  51.,  52.]])

"""
