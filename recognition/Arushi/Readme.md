# Problem:
Applying U-Net architecture to segment the ISICs data set with the UNet with all labels having a minimum Dice similarity coefficient of 0.7 on the test set.

# Dataset Description:
The dataset contains the largest collection of quality-controlled dermoscopic images of skin lesions. The dataset consisted of two folders Training input and Training GroundTruth which contained images of skin patches to detect the area that had melanoma. The input file dataset had dermascopic lesion images whereas the groundtruth file dataset had binary mask images that tell the location of skin patches. The 0 pixels in binary mask images represented the areas outside the skin lesion and 255 pixels represented the areas inside the skin lesion. The images in the dataset are of different sizes and varieties. The model aims to segment the test images to identify skin cancer. 

# Pre-processing Steps:
1.	The dataset path was asked from the user and the dataset was loaded with help of os library. 
2.	After the dataset was loaded, the images of the dataset were sorted with respect to labels in ascending order. 
3.	The images were captured using tqdm and were colour mode was set to greyscale. All the images in the dataset were resized to 256x256 size to have uniformity among all. 
4.	The model was then split into train, validation and test using sklearn train_test_split method. The ratio of the train, validation and test split is 60:20:20. First, we split the data into train and test with 80% and 20% and split the train data again into train and validation into 60% and 20%. A random state was used to reproduce the results and a higher weightage was given to the train test. 
5.	The input dataset images were divided by 255 to normalize the images and the segmented images were normalized to 0 and 1 label by remainder of 255 where 0 represented background of the image and 1 represented foreground of the image. 
6.	One hot encoding was performed to segmented dataset. 

# U-Net Architecture:
U-Net is a convolution network architecture that is used for precise and fast image segmentation. The architecture looks like a ‘U-shaped’ and majorly consists of two parts: encoder network and decoder network. The encoder(contracting path) is the left part of the diagram which is usually pre-trained classification network where convolution blocks and maxpool downsampling is applied to encode the input images into feature representations and the decoder(expansion path) is the right part of the diagram where upsampling and concatenation is performed by regular operations to get the features learned by encoder into dense classification. The basic concept of the model is to incorporate a ‘U’ shaped model in which the input images are first downsampled by decreasing their dimensions and then upsampling them back to their original scale. The model discovers and executes the segmentation process of the image during the course. This architecture consists of the following sections: contraction, bottleneck, expansion section and the skip connection.  
Below is the U-Net architecture image:

![U-Net architecture](https://github.com/arushi-mah/PatternFlow/blob/topic-recognition/recognition/Arushi/images/u-net_architecture.png)

In this model:
1.	The contracting path is composed of 4 blocks. It consists of two 3x3 convolutions with ReLu activation function and batch normalization and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step, we double the number of feature channels so that the architecture can learn the complex structures effectively. 
2.	The bottommost layer mediates between the contraction layer and the expansion layer. It uses two convolutional layers with batch normalization and a dropout layer and is followed by 2X2 up convolution layer. 
3.	The expansion section consists of 4 blocks. Each block consists of two 3x3 convolution layer with batch normalization, followed by a deconvolution layer with stride 2. After each block, number of feature maps used by convolution layer gets halved to maintain the symmetry. However, a concatenation layer is added to input every time which is appened by feature maps from contracting section. 
4.	At the end, the number of expansion blocks is the same as the number of contraction blocks. 

# Model Parameters and Training:
  - Adam optimizer and dice loss function are used when compiling the model. 
  - Callback parameter is being used to capture the best model.
  - 32 Batchsize is being used in this model.
  - 60 epochs is been used in this model.

# Pre-defined Packages:
- pandas
-	keras
-	tensorflow-gpu
-	tqdm
-	scikit-image
-	scikit-learn
-	matplotlib

# Usage of Files:
1.	Make sure that both the files Test_driver_script.py and ISIC_dataset_with_UNET.py are loaded in jupyter and kept in same directory. 
2.	Make sure all the pre-defined packages and libraries are installed and imported before the execution is carried out. 
3.	First, run all the code in ISIC_dataset_with_UNET.py file.  
4.	Then, run Test_driver_script.py file and while running the Test_driver_script.py script, the user command will ask to enter the path of training input and training groundtruth folder. 
5.	After running the Test_driver_script.py file, it will automatically compile all the code and give the desired results and plots and figures. 

# Model Output:
-	Accuracy plot with respect to Epochs for training and validation dataset with marker ‘x’ for best model.

![Accuracy plot](https://github.com/arushi-mah/PatternFlow/blob/topic-recognition/recognition/Arushi/images/accuracy_plot.png)

-	Dice loss plot with respect to Epochs for training and validation dataset with marker ‘x’ for best model.

![Dice loss plot](https://github.com/arushi-mah/PatternFlow/blob/topic-recognition/recognition/Arushi/images/loss_plot.png)

-	Overall dice score of the dataset: *0.8878574*

-	Plot input image, true image and predicted image

![final image](https://github.com/arushi-mah/PatternFlow/blob/topic-recognition/recognition/Arushi/images/final_output.JPG)


# Reference:
O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.

