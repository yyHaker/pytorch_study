# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

# build network - method 1 : 使用class继承torch中的神经网络结构
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))   # activation function for hidden layer
        x = self.predict(x)        # Liner output
        return x

net1 = Net(n_features=1, n_hidden=10, n_output=1)

# build netword -  method 2: easy and fast way
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
print("net1 architecture")
print(net1)
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
print("net2 architecture")
print(net2)
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""