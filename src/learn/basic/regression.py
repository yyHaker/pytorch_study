# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 建立数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor) shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())                  # noisy y data (tensor) shape=(100, 1)

# 用Variable来修饰这些数据tensor
x, y = Variable(x), Variable(y)

# 画图 （二维数据画法）
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 建立神经网络
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__功能
        # 定义每层用什么样的一种形式
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)    # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值， 神经网络分析输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性输出)
        x = self.predict(x)        # 输出值
        return x

# Net结构
net = Net(n_features=1, n_hidden=10, n_output=1)
print(net)

# 训练网络(并可视化)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # mean squared error

plt.ion()  # something about plotting

for t in range(1000):
    prediction = net(x)   # 喂给net训练数据x，输出预测值
    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
    print("step %d : %f" % (t, loss.data.numpy()))

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()           # 误差反向传播，计算参数更新值
    optimizer.step()          # 将参数更新值施加到net的parameters上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()






