# import pickle
# import gzip
# import dataset_util
# import data

# file_path_test = '/media/ravi/ubuntu_disk/ravi/atwork/other_repo/dgcnn/pytorch/data/nagoya_dataset_split/nagoya_dataset_split_train.pgz'

# def some(file_path_test):
#     with gzip.open(file_path_test, 'rb') as f:

#         someee = (pickle.load(f))
#     return someee['data']

# some(file_path_test)

# # print(dataset_util.load_pickle_file_with_label(file_path_test,compressed = True))


# # data.nagoya_dataset(partition='test', num_points=1024)



import torch


x = torch.tensor([1, 2, 3])
# x = x.repeat(4, 2)


print(x)

x = x.repeat(4, 2, 1)

print(x)
print(x.shape)