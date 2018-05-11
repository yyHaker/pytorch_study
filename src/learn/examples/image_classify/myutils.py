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
import pickle

np.random.seed(1)


def split_train_valid(list_csv, rate=0.9):
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
    valid_list = list_frame.loc[idx_valid, :]
    valid_list.to_csv('image_scene_data/valid_list.csv', index=False)
    train_list = list_frame.loc[idx_train, :]
    train_list.to_csv('image_scene_data/train_list.csv', index=False)
    print("save list done")


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


def write_data_to_file(data, path):
    """
    :param data: the data obj
    :param path: the store path
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_file(path):
    """
    :param path: the store path
    :return:
    """
    data_obj = None
    with open(path, 'rb') as f:
        data_obj = pickle.load(f)
    return data_obj


if __name__ == "__main__":
    split_train_valid(list_csv='image_scene_data/list.csv')
    print(len(os.listdir("image_scene_data/data/")))
