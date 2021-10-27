import tensorflow as tf
from data_preprocess import *
from improved_unet import *


home = "/home/tannishpage/Documents/COMP3710_DATA"
data_folder = "ISIC2018_Task1-2_Training_Input_x2/"
gt_folder = "ISIC2018_Task1_Training_GroundTruth_x2/"
"""
train, val, test = process_data_folders(os.path.join(home, data_folder))

move_files(train[0], os.path.join(home, data_folder),
            os.path.join(home, "train/data/images"))
move_files(train[1], os.path.join(home, gt_folder),
            os.path.join(home, "train/groundtruth/images"))

move_files(val[0], os.path.join(home, data_folder),
            os.path.join(home, "val/data/images"))
move_files(val[1], os.path.join(home, gt_folder),
            os.path.join(home, "val/groundtruth/images"))

move_files(test[0], os.path.join(home, data_folder),
            os.path.join(home, "test/data/images"))
move_files(test[1], os.path.join(home, gt_folder),
            os.path.join(home, "test/groundtruth/images"))
"""
train = create_generator(os.path.join(home, "train/data/images"),
                        os.path.join(home, "train/groundtruth/images"),
                        (128, 128), 750)

val = create_generator(os.path.join(home, "val/data/images"),
                        os.path.join(home, "val/groundtruth/images"),
                        (128, 128), 100)

test = create_generator(os.path.join(home, "test/data/images"),
                        os.path.join(home, "test/groundtruth/images"),
                        (128, 128), 250)



model = create_model((128, 128, 1))
#model.summary()

training_results = model.fit(train[0], train[1],
                            epochs=10,
                            validation_data=val,
                            batch_size=32)
