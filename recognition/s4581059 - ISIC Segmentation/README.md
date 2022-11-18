# 2017 ASIC Data Segmentation Using the Improved Unet Model
### Daniel Sayer - 45810590

## Improved Unet Algorithm
The improved unet algorithm (visualised in './imgs/Improved_Unet_Model.png' *sourced from [1]*) is an extension of the Unet algorithm. Both the Unet and the improved unet model are fully convolutional neural network, which aims to identify on an image where a certain item is located and draw a mask around it. In the ISIC data example this is outlining skin lesion so they can be assessed to wther they are a malignant melanoma, as if recognised and treated in early stages, it is readily curable [3]. 

The structure of all unet variations, consists of an encoder and a decoder (representing the downward and upward bend of the "U"), with concatenation layers adding corresponding layer of the encoder with the decoder. The improved Unet model differs significantly by incoroparing segmentation layers which are more specific convolutional layers, as they have a much smaller kernel, in this case (1,1). These layers are upscaled and added to the decoder to increase the accuracy of the unet model.

###### Notes:
    It is important to note the following about the unet structure:
        1. As it is a fully convolutional neural network it consists of no dense layers and only convolutions
        2. The final convolution of the decoder has only 2 filters and is of softmax activation
        3. All convolutional layers (except the final one) uses LeakyReLU activation

## How the algorithm works
The encoder section of the improved unet architecture works by condensing the information with each level and widening the receptive field. This is completed by doubling the number of channels (filters) with each level. In turn, this allows the network of observe complex relations which cannot be identified by humans. The decoder allows the model to highlight the location of the skin lesion. The decoder uses localisation layers to identify the location of the skin lesion, and the concatenation of the encoder to be more effiecient and accurate (as the encoder provides the "what we're looking for"). At each layer we upsample to recover the input size with the respective number of filters and back to the original image size. The segmentation layers provide a *deep supervison in the localization pathway [1]*, doing such is advantageous as for smaller training samples adding hidden layers, such as segmentation layers provide a good regularization for classification accuracy. For larger datasets this segmentation improves convergence behaviour [4]. A combination of the upscaled segmentation layers leads to a final softmax convolution allowing the model to outline and predict the location of a skin lesion, given an image of the skin (where applicable)

## Dependencies
    - Python 3.9
    - tensorflow: Version 2.1
    - random
    - numpy: Version 1.23
    - matplotlib: Version 3.6.1
    - glob
    - cv2: Version 3.4.4
    - os
    - sklearn: Version 1.1.2

## Appropiate Addresses
This project contains only one relevant address: the data set file address.
This address should be locally stored and must contain two subfolders named
    1. "ISIC-2017_Training_Data"
    2. "ISIC-2017_Training_Part1_GroundTruth"
This file uses the address: "C:/Users/danie/Downloads/ISIC DATA/" which fits the above criteria but this can be edited using the _path variable in the train.py file

## Training Validation Split
The training validation split method can be manipulated as the split ratio is a parameter to the method, however, a split of 0.2 was used (and is the default, if left unspecified) and thus 80% of the data was used for training. It is important to note the the split decimal is for equal parts validation and testing. This was used as this yielded good results in the training

## Results:
### Epoch Data Results
The training model uses 50 epoch to generate the model - the data for each can be found in epoch_data.txt
The first 5 and the last 5 epochs data can be found below:

    Epoch 1/50
    loss: 0.6256 - dice_similarity: 0.6508 - val_loss: 0.5168 - val_dice_similarity: 0.7049
    Epoch 2/50 
    loss: 0.4605 - dice_similarity: 0.7796 - val_loss: 0.4462 - val_dice_similarity: 0.7777
    Epoch 3/50
    loss: 0.4510 - dice_similarity: 0.7699 - val_loss: 0.3956 - val_dice_similarity: 0.7892
    Epoch 4/50 
    loss: 0.4519 - dice_similarity: 0.7713 - val_loss: 0.3969 - val_dice_similarity: 0.8153 
    Epoch 5/50 
    loss: 0.3791 - dice_similarity: 0.8116 - val_loss: 0.3200 - val_dice_similarity: 0.8494

    ...

    Epoch 46/50
    loss: 0.1481 - dice_similarity: 0.9350 - val_loss: 0.1293 - val_dice_similarity: 0.9453
    Epoch 47/50
    loss: 0.1207 - dice_similarity: 0.9446 - val_loss: 0.1285 - val_dice_similarity: 0.9459
    Epoch 48/50
    loss: 0.1119 - dice_similarity: 0.9503 - val_loss: 0.1291 - val_dice_similarity: 0.9447
    Epoch 49/50 
    loss: 0.1357 - dice_similarity: 0.9399 - val_loss: 0.1290 - val_dice_similarity: 0.9458 
    Epoch 50/50 
    loss: 0.1361 - dice_similarity: 0.9406 - val_loss: 0.1264 - val_dice_similarity: 0.9455

It is evident from the results above that the dice_similarity of the model well exceeds the target of 80% as it consistently holds and improves on 80% from epoch 5

###### Note: 
The data is not 100% consistent as it is a training set. And thus, attempting to reproduce results may result in very minor changes to the dice similarity at each epoch

### Loss Function Graph
For graph see ./imgs/loss_plot.png

### Dice Similarity Graph
For graph see ./imgs/dice_similarity_plot.png

### Mask Predictions
For mask comparison predictions see ./imgs/mask.png (shows 4 random samples)

## Example Use:
This repository contains the saved file for the model of which's data is shown to load the file. Add the follow to the predict.py file: 
    if __name__ == "__main__":
        saved_model = load_saved_model()
        saved_model.summary() <- will show the model summary
These can be run using the bash equivalent from terminal, however, will not be able to allow change to private variables such as _path

Analagously, to train the network, the following can be added to the train.py file:
    if __name__ == "__main__":
        train_model()

## Bibliography
[1] Fabian Isensee et al. (2018, February 28), Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge, available at: https://arxiv.org/pdf/1802.10508v1.pdf <br>
[2] Karan Jakhar (2019, October 19), Dice coefficient, IOU; for medium.com, available at: https://karan-jakhar.medium.com/100-days-of-code-day-7-84e4918cb72c <br>
[3] Angel Cummings, Nicholas Kurtansky (2017), Overview of the ISIC Collaboration, available at: https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/aboutIsicOverview <br>
[4] Chen-Yu Lee et al. (N.D) Deeply Supervised Nets, available at: https://pages.ucsd.edu/~ztu/publication/aistats15_dsn.pdf <br>