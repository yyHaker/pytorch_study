# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # calculate gradient

print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)

# computational graph
v_out.backward()   # backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)  # 初始Variable的梯度
"""
 0.5000  1.0000
 1.5000  2.0000
"""

# 获取Variable中的数据
print(variable)     # Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""


