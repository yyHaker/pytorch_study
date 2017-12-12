# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision   # database module

torch.manual_seed(1)   # reproducible

# Hyper parameters
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

# MNIST
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # save path or fetch path
    train=True,   # this is train data
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
# batch training 50examples, 1 channel, 28*28(50, 1,28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0, 1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000].cuda()/255.
test_y = test_data.test_labels[: 2000].cuda()

# Network architecture
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(            # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,        # input height
                out_channels=16,   # n_filters
                kernel_size=5,        # filter size
                stride=1,                  # filter movement/step
                padding=2                # 如果想要 conv2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),                # output shape (16, 28, 28)
            nn.ReLU(),   # activation
            nn.MaxPool2d(kernel_size=2)  # 在2x2的空间里面向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(   # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),                                   # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # output shape(32, 7, 7)
        )
        self.out = nn.Linear(32*7*7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成(batch_size, 32*7*7)
        output = self.out(x)
        return output, x      # return x for visualization

cnn = CNN()
cnn.cuda()         # move all parameters and buffers to the GPU
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999))
loss_func = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()   # Tensor on GPU
        b_y = Variable(y).cuda()    # Tensor on GPU

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| Step :', step,  '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)





