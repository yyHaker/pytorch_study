# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data tensor (100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data tensor (100, 1)
x, y = Variable(x), Variable(y)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()   # 误差反向传播，计算参数更新值
        optimizer.step()  # 将参数更新值施加到net的parameters上

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # two ways to save the net

    torch.save(net1, './model/net.pkl')  # save entire net
    torch.save(net1.state_dict(), './model/net_params.pkl')  # save only the parameters


def restore_net():
    # restore entire net1 to net2
    net2 = torch.load("./model/net.pkl")
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters to net3
    net3.load_state_dict(torch.load('./model/net_params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

save()
restore_net()
restore_params()
