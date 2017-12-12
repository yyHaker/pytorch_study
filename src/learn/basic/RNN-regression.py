# -*- coding: utf-8 -*-
"""
use sin to predict cos
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyper parameters
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 1  # rnn input size / image width
LR = 0.02
DOWNLOAD_MNIST = False

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,    # rnn hidden unit
            num_layers=1,        # 几层rnn layers
            batch_first=True   # # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        """
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        :param x:
        :param h_state:
        :return:
        """
        r_out, h_state = self.rnn(x, h_state)    # h_state 也要作为RNN的一个输入

        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)

# optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()     # continuously plot

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time steps
    # sin predict cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)   # shape(10, )
    y_np = np.cos(steps)

    # print(x_np[np.newaxis, :, np.newaxis].shape) # shape(1, 10, 1)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))  # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
    prediction, h_state = rnn(x, h_state)
    # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
    h_state = Variable(h_state.data)

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()


