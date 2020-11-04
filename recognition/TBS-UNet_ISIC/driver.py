import os

# Get filepaths
input_data_dir = "data\\reduced\\ISIC2018_Task1-2_Training_Input_x2"
gt_data_dir = "data\\reduced\\ISIC2018_Task1_Training_GroundTruth_x2"

input_data_paths = os.listdir(input_data_dir)
gt_data_paths = os.listdir(gt_data_dir)

print(input_data_paths)
print(gt_data_paths)