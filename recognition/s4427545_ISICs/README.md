# Using YOLOv5 to Identify Lesions in the ISIC Dataset

ISIC (International Skin Imaging Collaboration) released a dataset was released with the goal of reducing death from melanoma cancer by more easily and quickly identifying dangerous melanoma. I will refer to this datset simply as ISIC. The original ISICs data has a training, validation, and test set. But note that this repository only uses the training data and does its own splitting from that. ISIC has been releasing data for challenges every year over the last few years.

## YOLOv5 Overview


## Usage


## Data Processing
The data processing varied and evolved many times over the course of the project. At first I was just going to pass and images that had to mention 1024 x 1024, but I realised that six 640 x 640 was more appropriate. This is because YOLOv5 was designed for that image dimension. . I also attempted to augment the data using a few different tricks. There were five ways I augmented the data: rotation by 90°, rotation by 180°, rotation by 270°, a horizontal flip of the base image, and a vertical flip of the base image. Using all of these augmentations for each image in the training set was found to be too slow to train on, so I randomly chose two augmentations for each image with one of those also possibly being no augmentation. In the end, surprisingly this led to overfitting. Additionally, I was careful not to also augment the test or validation set data as this is completely inappropriate. Because the training said only included masks, I also had to generate the bounding boxes for each image as YOLO does not use masks. All the code that does the augmentations, as well as split the training set, can be found in isics_data_setup.py.

## Training Methodology
The training methodology was pretty simple. I took 20% of the training data as the test said, and took 20% of that 80% training set as the validation set. I initially took 10% of the data as the test set, as there is not a lot of images but I found 20% to still yield reasonable training performance. Also, three different versions of YOLOv5 were trained on. This included the small model, medium model, and the large model. These different models take a different amount of time to train and also differ in their memory consumption. Given the simplicity of the dataset, I expected the smallest model to still perform well which is useful to know as this could be run on a slow computer that does not have a GPU during inference time.

## Results
