# -*- coding: utf-8 -*-
"""
the image scene data manager.
(16623 images)
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

from myutils import split_train_valid, show_image_label, \
    show_image_label_batch


class ImageSceneData(Dataset):
    def __init__(self, categories_csv, list_csv, data_root, transform=None):
        """
        :param categories_csv: string,  path to teh categories csv.
        :param list_csv: string,  path to the list csv.
        :param data_root: string, directory with all the image.
        :param transform: (callable, optional), optional transform to be applied on
         a sample.
        """
        self.categories_frame = pd.read_csv(categories_csv)
        self.list_frame = pd.read_csv(list_csv)
        self.data_root = data_root
        self.transform = transform

        self.id2tag = {}
        self.build_id2tag()

    def __len__(self):
        return len(self.list_frame)

    def __getitem__(self, idx):
        """通过idx索引到图像"""
        img_name = os.path.join(self.data_root, self.list_frame.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        # convert to same channel(RGB)
        image = image.convert("RGB")
        label = self.list_frame.iloc[idx, 1]
        sample = {"image": image, "label": label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def build_id2tag(self):
        for i in range(20):
            self.id2tag[i] = self.categories_frame.iloc[i, 2]

    def analyze_data(self):
        size = len(self.list_frame)
        count_dict = {}
        for i in range(20):
            count_dict[self.id2tag[i]] = 0
        for i in range(size):
            tag_name = self.id2tag[self.list_frame.iloc[i, 1]]
            count_dict[tag_name] += 1
        return count_dict


class ImageSceneTestData(Dataset):
    def __init__(self, categories_csv, list_csv, data_root, transform=None):
        """test data dataset
        :param categories_csv: string,  path to teh categories csv.
        :param list_csv: string,  path to the list csv.
        :param data_root: string, directory with all the image.
        :param transform: (callable, optional), optional transform to be applied on
         a sample.
        """
        self.categories_frame = pd.read_csv(categories_csv)
        self.list_frame = pd.read_csv(list_csv)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.list_frame)

    def __getitem__(self, idx):
        """通过idx索引到图像"""
        img_name = os.path.join(self.data_root, self.list_frame.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        # convert to same channel(RGB)
        image = image.convert("RGB")
        sample = {"image": image}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


if __name__ == "__main__":
    # transform
    data_transform = transforms.Compose([
        transforms.Resize((random.randint(256, 480), random.randint(256, 480))),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(random.randint(1, 90)),
        transforms.RandomRotation(random.randint(1, 90)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # use Dataset
    image_data = ImageSceneData(categories_csv='image_scene_data/categories.csv',
                                list_csv='image_scene_data/list.csv',
                                data_root='image_scene_data/data',
                                transform=data_transform)
    # use DataLoader
    image_data_loader = DataLoader(image_data, batch_size=4, shuffle=True,
                                   num_workers=4)
    for i_batch, sampled_batch in enumerate(image_data_loader):
        # print(len(sampled_batch['image']))
        print(i_batch, sampled_batch['image'].size(), sampled_batch['label'].size())

        # observe 4th batch and stop
        if i_batch <= 5:
            plt.figure()
            show_image_label_batch(sampled_batch, image_data.id2tag)
            plt.axis('off')
            plt.ioff()
            plt.show()
            plt.pause(1)
