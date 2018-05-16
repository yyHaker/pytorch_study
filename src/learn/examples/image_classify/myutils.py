# -*- coding: utf-8 -*-
"""
my utils
"""
from torchvision import utils
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import random
from PIL import Image
import skimage
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
    print("train size: {}/{}, valid size: {}/{}".format(train_size,
                                                        data_size, data_size-train_size, data_size))
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


def pca_Jittering(img):
    """
    Apply pca jittering  to the PIL img.
    :param img:
    :return:
    """
    img = np.asarray(img, dtype='float32')
    img /= 255
    img_size = img.size // 3
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(img_cov)
    p = np.transpose(p)
    alpha1 = random.normalvariate(0, 0.3)
    alpha2 = random.normalvariate(0, 0.3)
    alpha3 = random.normalvariate(0, 0.3)
    v = np.transpose(
        (alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
    add_num = np.dot(p, v)
    img2 = np.array(
        [img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    img2 *= 255
    img2 = Image.fromarray(np.uint8(img2))

    return img2


def random_noise(img, p=0.1):
    a = random.random()
    if a < p:
        img = np.asarray(img, dtype='float32')
        img /= 255
        img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)
        img *= 255
        img = Image.fromarray(np.uint8(img))
    return img


if __name__ == "__main__":
    split_train_valid(list_csv='image_scene_data/list.csv')
    print("root data image: {}".format(len(os.listdir("image_scene_data/data/"))))
    # print(random.sample([224, 256, 384, 480, 640], 1))
