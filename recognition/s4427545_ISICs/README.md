# Using YOLOv5 to Identify Lesions in the ISIC Dataset
ISIC (International Skin Imaging Collaboration) released a dataset with the goal of reducing death from melanoma cancer by more easily and quickly identifying dangerous melanoma through the communities creation of machine learning models. I will refer to this datset simply as ISIC. The original ISICs data has a training, validation, and test set. But note that this repository only uses the training data and does its own splitting from that. ISIC has been releasing data for challenges every year over the last few years.

## YOLOv5 Overview


## Usage
For the script that generates the validation set, the test set, and the augmented training data, there are three runtime parameters. This script is called isics_data_setup.py. The parameters are (in order), the directory of the dataset, the validation split of the training data, and the test split.
Note that within the directory containing the dataset, the data must be structured as follows:
<img width="431" alt="Screen Shot 2021-11-20 at 7 00 07 am" src="https://user-images.githubusercontent.com/43590948/142691136-8cbbcce7-0f8a-4640-abc5-0ffe4906eb54.png">.
Additionally, in the current working directory there should be a folder named _datasets_ with a subfolder named _ISIC_, and then three sub folders within that name and training, validation, and test. Then finally, within each of those there should be a folder named images and labels. E.g.
<img width="443" alt="Screen Shot 2021-11-20 at 7 03 50 am" src="https://user-images.githubusercontent.com/43590948/142691418-7fef2354-bc8a-4575-863e-b354feea827b.png">

For the driver script (driver.py), there are 5 runtime parameters:
batch_size, mode (training or predict), epochs, data (.yaml file), model_type (.pt file). The .pt file can be found on the official YOLOv5 GitHub page, as well as an example YAML file. I included one in the repository.

## Data Processing
The data processing varied and evolved many times over the course of the project. At first I was just going to use the original images that had dimension 1024 x 1024, but I realised that 640 x 640 was more appropriate. This is because YOLOv5 was designed for that image dimension. I also attempted to augment the data using a few different tricks. There were five ways I augmented the data: rotation by 90°, rotation by 180°, rotation by 270°, a horizontal flip of the base image, and a vertical flip of the base image. Using all of these augmentations for each image in the training set was found to be too slow to train on, so I randomly chose two augmentations for each image with one of those also possibly being no augmentation. In the end, this surprisingly led to overfitting. Additionally, I was careful not to also augment the test or validation set data as this is inappropriate. Because the training said only included masks, I also had to generate the bounding boxes for each image, as YOLO does not use masks. All the code that does the augmentations, as well as split the training set, can be found in isics_data_setup.py.

## Training Methodology
The training methodology was kept simple. I took 20% of the training data as the test set, and took 20% of that 80% training set as the validation set. I initially took 10% of the data as the test set, as there is not a lot of images, but I found 20% to still yield reasonable training performance. Also, three different versions of YOLOv5 were trained on. This included the small model, medium model, and the large model. These different models take a different amount of time to train and also differ in their memory consumption. Given the simplicity of the dataset, I expected the smallest model to still perform well which is useful to know as this could be run on a slow computer that does not have a GPU during inference time. Additionally, I expected the smaller model to do better given the simplicity of the data. This is because models with less parameters are less prone to overfitting on simple data, versus a larger model. For example, YOLOv5 can be used for images containing multiple objects from up to many different classes. But, this dataset only has a single class with one instance per image.

## Results
