import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

original_data_dir = "dgcnn/pytorch/data/nagoya_dataset"
splited_data_dir = "dgcnn/pytorch/data/nagoya_dataset_split"

print("Splitting dataset...")
splitfolders.ratio(original_data_dir, output = splited_data_dir, seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values
print("Done")