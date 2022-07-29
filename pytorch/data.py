#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob


import gzip
import numpy as np
import pickle
from torch.utils.data import Dataset
import dataset_util



# def translate_pointcloud(pointcloud): What is this translation for ?
#     xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
#     translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
#     return translated_pointcloud


# def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02): # adding noise to the pointcloud
#     N, C = pointcloud.shape
#     pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
#     return pointcloud


def load_data(partition):

    cluster = True

    if cluster:

        file_path_train = '/scratch/rselva2s/bit-bots/dgcnn/pytorch/data/nagoya_split_data/nagoya_dataset_split_train.pgz'
        file_path_test = '/scratch/rselva2s/bit-bots/dgcnn/pytorch/data/nagoya_split_data/nagoya_dataset_split_test.pgz'

    else:

        file_path_train = '/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset_split/nagoya_dataset_split_train.pgz'
        file_path_test = '/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset_split/nagoya_dataset_split_test.pgz'

    if partition == 'train':
        
        data, label = dataset_util.load_pickle_file_with_label(file_path_train,compressed = True)

        return data, label
        
    elif partition == 'test':
        
        data, label = dataset_util.load_pickle_file_with_label(file_path_test,compressed = True)
        
        return data, label

class nagoya_dataset(Dataset):

    def __init__(self, num_points, partition='train'):

        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):            # why only training data is translated?

        pointcloud = self.data[item][:self.num_points]

        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):

        return self.data.shape[0]


if __name__ == '__main__':

    train = nagoya_dataset(1024)
    test = nagoya_dataset(1024, 'test')

    for data, label in train:
        print(data)
        print(label)

        print("-----")


