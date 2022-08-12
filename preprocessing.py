from ctypes import util
import numpy as np
import os
from util import split_data


split_data( source_dir="/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset",
            destination_dir="/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset_split"
            , ratio = (.8, .2) )

dataset_dir = "/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset_split"
dataset_name = "nagoya_dataset_split"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
all_dataset = [train_dir, test_dir]

# downsample and padding param
# downsample if > 2048, and pad if < 2048
num_points = 2048

# split train or test files into multiple files
number_train_file = 4
number_test_file = 1

total_dataset = 0

for split in ["train", "test"]:
    data_path = os.path.join(dataset_dir, split)
    train_classes = [label for label in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, label))]
    train_classes.sort()
    # generate labels into a txt file 
    with open(os.path.join(dataset_dir, split + "_labels.txt"), "w") as f:
        for e, lab in enumerate(train_classes):
            f.write(str(e) +' - '+ str(lab) + "\n")

    print("Dataset: ", split)
    data = []
    labels = []
    for i,label in enumerate (train_classes):
        pcd_files = os.listdir(os.path.join(data_path, label))
        np.random.shuffle(pcd_files)
        print ("Preprocessing ",label)
        data_per_class = []
        total_per_class = 0
        for j, pcd_file_name in enumerate(pcd_files):
            pcd_path = os.path.join(data_path, label, pcd_file_name)
            xyzrgb = util.extract_pcd(pcd_path, num_points=num_points, 
                                      color=True, downsample_cloud=True, 
                                      pad_cloud=True, normalize_cloud=True)
            
            if xyzrgb is not None:
                labels.append(i)
                data.append(xyzrgb)

                total_dataset += 1
                total_per_class += 1
            
    data = np.asarray(data)
    labels = np.asarray(labels)
    print ("Data", data.shape)
    print ("Labels", labels.shape)
    
    #shuffle dataset
    data, labels = dataset_util.randomize(data, labels)
    
    data_dict = {}
    data_dict['data'] = data
    data_dict['labels'] = labels
    
    # save pickle file
    dataset_util.save_dataset_and_compress(data_dict, dataset_dir+'/{}_{}'.format(dataset_name, split))  
    
    data = []
    labels = []
    data_dict.clear()