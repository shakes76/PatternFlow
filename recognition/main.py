"""
Classify laterality of  the OAI AKOA with minimum accuracy of 0.9
@author : Wangwenhao(46039941)
"""
import os
import shutil
import numpy as np

# split the original image into train,test,val

files = np.array(os.listdir("AKOA_Analysis"))
index = [i for i in range(len(files))]
np.random.shuffle(index)
train_end = int(len(files) * 0.7)
val_end = int(len(files) * 0.9)
train_files = files[:train_end].tolist()
val_files = files[train_end:val_end].tolist()
test_files = files[val_end:].tolist()

# copy and split the data into left and right part according to their name

old_dir = "AKOA_Analysis"
train_right_dir = "train/right"
train_left_dir = "train/left"
val_right_dir = "val/right"
val_left_dir = "val/left"
test_right_dir = "test/right"
test_left_dir = "test/left"
for file in train_files:
	old_file_path = os.path.join(old_dir, file)
	if file.split("_WE_")[1][0] == 'R':
		new_file_path = os.path.join(train_right_dir, file)
	if file.split("_WE_")[1][0] == 'L':
		new_file_path = os.path.join(train_left_dir, file)
	shutil.copy(old_file_path, new_file_path)

for file in val_files:
	old_file_path = os.path.join(old_dir, file)
	if file.split("_WE_")[1][0] == 'R':
		new_file_path = os.path.join(val_right_dir, file)
	if file.split("_WE_")[1][0] == 'L':
		new_file_path = os.path.join(val_left_dir, file)
	shutil.copy(old_file_path, new_file_path)

for file in test_files:
	old_file_path = os.path.join(old_dir, file)
	if file.split("_WE_")[1][0] == 'R':
		new_file_path = os.path.join(test_right_dir, file)
	if file.split("_WE_")[1][0] == 'L':
		new_file_path = os.path.join(test_left_dir, file)
	shutil.copy(old_file_path, new_file_path)


