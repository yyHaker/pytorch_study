# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from models import LeNet5

USE_CUDA = torch.cuda.is_available()

# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.
                               Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./cifiar', train=True,
                                        transform=transform, download=True)
trainloader = Data.DataLoader(trainset, batch_size=4,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifiar', train=False,
                                       transform=transform, download=True)
testLoader = Data.DataLoader(testset, batch_size=4,
                             shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


# show some image
def imgshow(img):
    img = img / 2 + 0.5  # unnormalize
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# mgshow(torchvision.utils.make_grid(images))

# define models
net = LeNet5()
if USE_CUDA:
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# train
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if USE_CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
    # every epoch test the model
    correct = 0
    total = 0
    for data in testLoader:
        images, labels = data
        images = Variable(images)
        if USE_CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        if USE_CUDA:
            predicted = predicted.cpu().data.numpy()
        else:
            predicted = predicted.numpy()
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("epoch %d , Accuracy of the net on the 1000 test images: %d %% "
          % (epoch, 100 * correct / total))

print("training is done!")


