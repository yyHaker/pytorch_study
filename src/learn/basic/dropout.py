# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

N_SAMPLES = 20
N_HIDDEN = 300

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

# testing data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
test_x, test_y = Variable(test_x,requires_grad=False), Variable(test_y, requires_grad=False)

# show data
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label="train")
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label="test")
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# build the network net_overfitting and net_droped
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1)
)

net_droped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1)
)

optimizer_overfit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_dopout = torch.optim.Adam(net_droped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

# plotting
plt.ion()

for t in range(500):
    pred_overfit = net_overfitting(x)
    pred_drop = net_droped(x)

    loss_overfit = loss_func(pred_overfit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_overfit.zero_grad()
    optimizer_dopout.zero_grad()
    loss_overfit.backward()
    loss_drop.backward()
    optimizer_overfit.step()
    optimizer_dopout.step()

    if t % 10 == 0:
        # 将神经网络转换成测试形式， 画好图之后改回训练形式(因为drop网络在训练和测试时参数不一样)
        net_overfitting.eval()
        net_droped.eval()

        # plotting
        plt.cla()

        test_pred_overfit = net_overfitting(test_x)
        test_pred_drop = net_droped(test_x)

        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label="train")
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label="test")
        plt.plot(test_x.data.numpy(), test_pred_overfit.data.numpy(), 'r--', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label="dropout(50%)")
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_overfit, test_y).data[0],
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data[0],
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        # 将两个网路改回训练形式
        net_overfitting.train()
        net_droped.train()
plt.ioff()
plt.show()