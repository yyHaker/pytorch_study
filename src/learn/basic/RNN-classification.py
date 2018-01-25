# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time

# 设置全局cudnn不可用
torch.backends.cudnn.enabled = False

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
TIME_STEP = 28   # rnn 时间步数/图片高度
INPUT_SIZE = 28  # rnn每步输入值/图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = True

# mnist
train_data = torchvision.datasets.MNIST(
    root='./mnist/',        # save or fetch path
    train=True,      # this is training data
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
# plot one example
print(train_data.train_data.size())   # (60000, 28, 28)
print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor)
# shape (2000, 28, 28) value in range(0, 1)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000].cuda() / 255.
test_y = test_data.test_labels[:2000].cuda()   # convert to numpy array

# RNN architecture
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,   # 图片每行的数据像素点
            hidden_size=64,  # rnn hidden unit
            num_layers=1,    # 有几层rnn layers
            batch_first=True  # input & output 会是以batch_size为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)  # 输出层

    def forward(self, x):
        """
         x shape (batch, time_step, input_size)
         r_out shape (batch, time_step, output_size)
         h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
         h_c shape (n_layers, batch, hidden_size)
        :param x:
        :return:
        """
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示hidden state 会用全0的state
        # 选取最后一个时间点的r_out输出
        # 这里r_out[:, -1, ;]的值也是h_n的值
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
rnn.cuda()
print(rnn)

start_time = time.time()
# training
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # print(x)  # 64x1x28x28
        # print(y)  # 64
        b_x = Variable(x.view(-1, 28, 28)).cuda()  # reshape x to (batch, time_step, input_size)
        b_y = Variable(y).cuda()

        output = rnn(b_x)   # rnn output
        loss = loss_func(output, b_y)
        optimizer.zero_grad()    # clear gradients for this training step
        loss.backward()              # backpropagation, compute gradients
        optimizer.step()             # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)    # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| Step: ', step, '| Train loss: %.4f' %
                  loss.data[0], '| Test accuracy %.2f' % accuracy)
end_time = time.time()
print("total cost time:", end_time - start_time)
# not cudnn total cost time: 478.00118017196655

