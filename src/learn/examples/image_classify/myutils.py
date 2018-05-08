# -*- coding: utf-8 -*-
"""
my utils
"""
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


def split_train_valid(data_root, list_csv, rate=0.9):
    """split data to train and valid.
    Actually change list.csv to train_list.csv and valid_list.csv.
    """
    list_frame = pd.read_csv(list_csv)
    data_size = len(list_frame)
    train_size = int(data_size * rate)
    idx_list = np.arange(data_size)
    np.random.shuffle(idx_list)
    idx_train = idx_list[: train_size]
    idx_valid = idx_list[train_size+1:]
    print(idx_valid)

def show_image_label(image, label):
    """show image and label"""
    plt.imshow(image)
    plt.title(label)
    plt.pause(1.0)


def show_image_label_batch(sampled_batch, id2tag):
    """show image with label for a batch of example"""
    images_batch, labels_batch = sampled_batch['image'], sampled_batch['label']
    batch_size = len(images_batch)
    img_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        # print(labels_batch[i].numpy(), type(labels_batch[i].numpy()))
        id = int(labels_batch[i].numpy())
        plt.text(i * img_size, 0, id2tag[id])


if __name__ == "__main__":
    split_train_valid(data_root='image_scene_data/data', list_csv='image_scene_data/list.csv')
