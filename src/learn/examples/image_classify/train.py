# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import AlexNet
from dataUtils import ImageSceneData

USE_CUDA = torch.cuda.is_available()

# load data
# transform
train_data_transform = transforms.Compose([
    transforms.RandomSizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# use Dataset
train_image_data = ImageSceneData(categories_csv='image_scene_data/categories.csv',
                                  list_csv='image_scene_data/train_list.csv',
                                  data_root='image_scene_data/data',
                                  transform=train_data_transform)
# use DataLoader
train_image_data_loader = DataLoader(train_image_data, batch_size=4, shuffle=True,
                                     num_workers=4)

# transform
valid_data_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.ToTensor(),
])
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# use Dataset
valid_image_data = ImageSceneData(categories_csv='image_scene_data/categories.csv',
                                  list_csv='image_scene_data/valid_list.csv',
                                  data_root='image_scene_data/data',
                                  transform=valid_data_transform)
# use DataLoader
valid_image_data_loader = DataLoader(valid_image_data, batch_size=4, shuffle=True,
                                     num_workers=4)

# define models
net = AlexNet()
if USE_CUDA:
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# train
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_image_data_loader, 0):
        inputs, labels = data['image'], data['label']
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
        if i % 1000 == 999:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 1000))
            running_loss = 0.0
    # every epoch test the model on the valid data
    correct = 0
    total = 0
    for data in valid_image_data_loader:
        images, labels = data['image'], data['label']
        images = Variable(images)
        if USE_CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        if USE_CUDA:
            predicted = predicted.cpu().numpy()
        else:
            predicted = predicted.numpy()
        total += labels.size(0)
        labels = labels.numpy()
        correct += (predicted == labels).sum()
    print("epoch %d , Accuracy of the net on the 1000 valid images: %d %% "
          % (epoch, 100 * correct / total))

print("training is done!")


