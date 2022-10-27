# Yeheng_Sun_S4548085_COMP3710_ADNIClassifier

## The code is written to provide a solution for problem 6. The detailed description is listed below:

# Problem Defination

Classify Alzheimer’s disease (normal and AD) of the ADNI brain dataset using a visual transformer.

# Algorithm Description(Concept)

The model consists of three main modules. The first module is the Patch Encoders module. It consists of a patch layer which turns images into patches, followed by a patch encoder layer which encodes patches into vectors. The second is the transformer module, this module measure the relationships between pairs of input vectors. Finally, there is a multi-layer perceptron module which has 2 layers with 1024 x 512 neurons that act as a classifier. Combine all the modules together and they make up the Vision Transformer model.

# Train Procedure

The original image is 256 x 240 pixels. I resize the image into 256 X 256 pixels such that the input shape of the model is (256, 256, 3) in which 3 stands for RGB. The metric I used is Binary Accuracy, and the loss function I used is Binary Cross-entropy. The number of heads of the multi-head attension layer is 4. I used Adam optimizer and I set the learning rate to .0003 with the ReduceLROnPlateau function implemented such that when the loss remains unchanged, the learning rate will be reduced.

I trained the model through 100 ephocs.

The dataset had already been divided into training, validation, and test sets. The validation dataset is useful during training to monitor for overfitting, and the test dataset was used to assess model generalisation capability on a set not seen during training.


# Structure of the project

**“README.MD”** is a file you are reading, provide detailed description of the dataset and scripts. 

**ADNI_AD_NC_2D** is a directory containing the ADNI brain image dataset, the directories structure and corresponding descriptions are shown below:

```
ADNI_AD_NC_2D  #　a directory containing the ADNI brain image dataset 
└─AD_NC        # a sub-directroy 
    ├─test     # a directory containing the test set 
    │  ├─AD    # a directory containing all the brain images with Alzheimer’s disease in the test set 
    │  └─NC    # a directory containing all cognitively normal brain images in the test set 
    └─train    # a directory containing the train set 
        ├─AD   # a directory containing all the brain images with Alzheimer’s disease in the train set 
        └─NC   # a directory containing all cognitively normal brain images in the train set 
```

**requirments.txt** is a txt file containing all required dependencies for a specific version

**“dataset.py"** containing functions for loading train and test images, the dataset directory 'DATADIR_train' and 'DATADIR_test' should be modified with proper paths

**“modules.py"** containing the source code of the visual transformer, including the implementaion of 'Patches' and 'PatchEncoder' class, 'mlp' and 'create_vit_classifier' funtion. 

- 'Patches' class is used to split a raw imag into patches
- 'PatchEncoder' class is used to encode patches into vectors
- 'mlp' is the implementaion of multilayer perceptron, which place within the visual transformer
- 'create_vit_classifier' is the implementaion of visual transformer itself

**“train.py"** containing the source code for model training, to ensure reproduciblility, please remain all the parameters unchange. 'run_experiment' is the function defining optimizer and checkpoint of the model, while the training process of the model is implemented within.

**“predict.py"** containing source code for printing and ploting model performance in test set.


# Preprocesssing
The original dataset have train and test directories. The images in train directory is for training and validation. The images in test directory are for testing. I split the images in train directory into train data and validation data with the ratio 7:3 respectively.

# Changes compared to original ViT model
In module.py, I remove data augmentation layers such that original image data (256, 256, 3) will be directly split into patches and encoded into vectors. In this setting, it will increase the train accuracy and result in an acceptable test accuracy. In addition, I increase the patch size and reduce the number of patches, as the number of patches is equal to (image_size // patch_len) ** 2, such that the number of patches is reduced from 400+ to 20+. Furthermore, I implemented the ReduceLROnPlateau function such that when the loss remains unchanged, the learning rate will be reduced. Finally, to keep the model from becoming too complicated, I reduce the neurons in the MLP layers from \[2048,1024\] to \[1024,512\].

# Experiment Reproducible Step
- Download dataset from https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download and unzip it into the root directory.
- Install all the dependencies in requirements.txt.
- Make sure you cd to the root directory and the structure of dataset directory is ./ADNI_AD_NC_2D/AD_NC/.
- Run the train.py file to start training, after the training run the predict.py file to evalute the model.

# Output
After train and save the weights of the model, we are able to use it for prediction. Load the file "model.h5", the corresponding model weights will be loaded into the model. We use the test set for evalution and having 83.27% accuracy.

### Train Accuracy
![accuracy.png](https://i.postimg.cc/T3MqvL7w/accuracy.png)

### Train Loss
![loss.png](https://i.postimg.cc/tC5ZJ1YM/loss.png)


# Example of Dataset Image

### Alzheimer’s disease image
![218391-78.jpg](https://i.postimg.cc/2jLcjkKh/218391-78.jpg)

### Normal image
![808819-88.jpg](https://i.postimg.cc/Gh8CCnk5/808819-88.jpg)