# -*- coding: utf-8 -*-
"""
some models to train the image_scene_data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """the fake AlexNet models"""
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=20)

    def forward(self, x):
        """
        :param x: input tensor, [None, 3x227x227]
        :return:
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))  # 96x27x27
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))  # 256x13x13
        x = F.relu(self.conv3(x))  # 383x13x13
        x = F.relu(self.conv4(x))  # 256x13x13
        x = F.max_pool2d(x, kernel_size=(2, 2))  # 256x6x6
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

