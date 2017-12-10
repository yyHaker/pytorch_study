# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating 200x2
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer (200, )

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

# plotting（三维数据画法）
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_features=2, n_hidden=10, n_output=2)  # 2 classes
print(net)

# training
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
lossfunc = torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(1000):
    out = net(x)  # 喂给net x， 输出分析值
    loss = lossfunc(out, y)  # 计算loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        print("step %i: loss %f" % (t, loss.data.numpy()))
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]  # (200, )
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()













