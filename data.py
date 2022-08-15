#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""




import numpy as np
from torch.utils.data import Dataset
import util
from util import extract_pcd


def translate_pointcloud(pointcloud): #What is this translation for ?
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02): # adding noise to the pointcloud
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def load_data(partition):

    cluster = False

    if cluster:

        file_path_train = 'dataset/nagoya_split_data/nagoya_dataset_split_train.pgz'
        file_path_test = 'dataset/nagoya_split_data/nagoya_dataset_split_test.pgz'

    else:

        
        file_path_train = 'dataset/nagoya_dataset_split/nagoya_dataset_split_train.pgz'
        file_path_test = 'dataset/nagoya_dataset_split/nagoya_dataset_split_test.pgz'



    if partition == 'train':
        
        data, label = util.load_pickle_file_with_label(file_path_train,compressed = True)

        return data, label
        
    elif partition == 'test':
        
        data, label = util.load_pickle_file_with_label(file_path_test,compressed = True)
        
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
            # pointcloud = translate_pointcloud(pointcloud) # add translation to the pointcloud 
            # pointcloud = jitter_pointcloud(pointcloud) # adding noise to the pointcloud
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):

        return self.data.shape[0]


class infer_data(Dataset):

    def __init__(self, num_points, pcl_path, transform=None):

        self.pcl_path = pcl_path
        self.num_points = num_points
        self.data = None

    def __len__(self):
        return 1000

    def __getitem__(self, x):
        
        self.data = extract_pcd(self.pcl_path,num_points=self.num_points)

        label = np.array([0])

        return self.data, label


if __name__ == '__main__':

    # train = nagoya_dataset(1024)
    # test = nagoya_dataset(1024, 'test')

    test = infer_data(1024, '/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/pretrained/11.pcd')
    print(test)
