import matplotlib.pyplot as plt
import numpy as np
import gzip
import os
import argparse
import random
import time


# 读取图像数据，展成一维数列
def readimage(path):
    with gzip.open(path, 'rb') as f:
        img = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    return img


# 读取标签数据
def readlabel(path):
    with gzip.open(path, 'rb') as f:
        lab = np.frombuffer(f.read(), np.uint8, offset=8)
    return lab


# 获得包含每种标签的全部集合
def get_dataset(train):  # 如果train=True，返回训练集，否则返回测试集
    data_path = "/home/shiliangliang/MNIST"  # put in your MNIST's data path
    data = []
    label = []
    if train:
        data = np.array(readimage(os.path.join(data_path, "train-images-idx3-ubyte.gz")) / 255, dtype=float)
        # print("train_data", train_data.shape)
        label = np.array(readlabel(os.path.join(data_path, "train-labels-idx1-ubyte.gz")), dtype=int)
        # print("train_label", train_label.shape)
    else:
        data = np.array(readimage(os.path.join(data_path, "t10k-images-idx3-ubyte.gz")) / 255, dtype=float)
        label = np.array(readlabel(os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")), dtype=int)

    data_new = []
    label_new = []

    for i in range(10):
        index = np.where(label == i)
        data_i = data[index]
        label_i = label[index]
        data_new.append(data_i)
        label_new.append(label_i)

    # print(len(data))
    return data_new, label_new


def config():
    parser = argparse.ArgumentParser(description='Train Multiplex')
    parser.add_argument('--validate_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, help='Num of one batch')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Num of epochs')
    args = parser.parse_args()
    return args


# 获得训练集每类前500， 测试集每类前100
def load(numList):
    train_data, train_label = get_dataset(train=True)
    # print(len(train_data), len(train_label))
    test_data, test_label = get_dataset(train=False)
    # print(len(test_data), len(test_label))

    train_img = np.zeros((numList[0], 784))
    train_tag = np.zeros((numList[0],))
    test_img = np.zeros((500, 784))
    test_tag = np.zeros((500,))
    for i in range(10):
        train_d = train_data[i][0:numList[i]]
        train_l = train_label[i][0:numList[i]]
        # print(train_d.shape, train_l.shape)
        test_d = test_data[i][0:100]
        test_l = test_label[i][0:100]
        # print(test_d.shape, test_l.shape)
        if i == 0:
            train_img = train_d
            train_tag = train_l
            test_img = test_d
            test_tag = test_l
        else:
            train_img = np.vstack((train_img, train_d))
            train_tag = np.hstack((train_tag, train_l))
            test_img = np.vstack((test_img, test_d))
            test_tag = np.hstack((test_tag, test_l))

    # print(train_img.shape)
    # print(test_img.shape)
    # print(np.sum(np.array(numList)))

    train_img = np.array(train_img, dtype=float).reshape((-1, 28 * 28))
    train_tag = np.array(train_tag).reshape((-1))
    test_img = np.array(test_img, dtype=float).reshape((-1, 28 * 28))
    test_tag = np.array(test_tag).reshape((-1))

    return train_img, train_tag, test_img, test_tag



