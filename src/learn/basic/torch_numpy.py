# -*- coding: utf-8 -*-
"""
view more https://morvanzhou.github.io/tutorials/machine-learning/torch/2-01-torch-numpy/
"""
import torch
import numpy as np

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array', np_data,
    '\ntorch_tensor', torch_data,
    '\ntensor to array', tensor2array
)

# abs
data = [-1, -2, -1, -2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy:', np.abs(data),
    '\ntorch', torch.abs(tensor)  # torch.FloatTensor of size 4
)

# sin
print(
    '\nsin',
    '\nnumpy:', np.sin(data),
    '\ntorch', torch.sin(tensor)
)

# mean
print(
    '\nmean',
    '\nnumpy', np.mean(data),
    '\ntorch', torch.mean(tensor)
)

# matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
# correct method
print(
    '\nmatrix multiplication(matmul)',
    '\nnumpy', np.matmul(data, data),
    '\ntorch', torch.mm(tensor, tensor)  # torch.FloatTensor of size 2x2
)

