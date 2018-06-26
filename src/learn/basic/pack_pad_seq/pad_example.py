# -*- coding: utf-8 -*-
"""
understand the use of pack_padded_sequence and pad_packed_sequence
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size = 3
max_length = 3
hidden_size = 2
n_layers = 1
num_features = 1
# container
batch_in = torch.zeros((batch_size, max_length, num_features))

# data
vec_1 = torch.FloatTensor([[1], [2], [3]])
vec_2 = torch.FloatTensor([[4], [5], [0]])
vec_3 = torch.FloatTensor([[6], [0], [0]])

batch_in[0] = vec_1
batch_in[1] = vec_2
batch_in[2] = vec_3

batch_in = Variable(batch_in)   # 3x3x1
seq_lengths = [3, 2, 1]  # list of integers holding information about the batch size at each sequence step

# pack it(Packs a Tensor containing padded sequences of variable length)
pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
print(pack)   # [6, 1]
"""
>>> pack
PackedSequence(data=tensor([[ 1.],
        [ 4.],
        [ 6.],
        [ 2.],
        [ 5.],
        [ 3.]]), batch_sizes=tensor([ 3,  2,  1]))
"""
# retrieve the original sequence back if I do
pad = torch.nn.utils.rnn.pad_packed_sequence(pack, [3, 2, 1])
print(pad)
"""
>>> pad
(tensor([[[ 1.],
         [ 2.],
         [ 3.]],

        [[ 4.],
         [ 5.],
         [ 0.]],

        [[ 6.],
         [ 0.],
         [ 0.]]]), tensor([ 3,  2,  1]))
"""




