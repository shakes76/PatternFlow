# Report - Pattern Recognition
# Objective:
The agenda of this Project is to Apply Improved U-Net [1] architecture on the ISICs data set by predicting the segmentation responses on lesion images in test data set with an overall minimum Dice Score of `0.8`.
The Data Set consist of the "Skin Lesion Analysis Towards Melanoma Detection" the challenge leverages a dataset of annotated skin lesion images from the ISIC Archive. The dataset contains both raw (Input data) and segmented (Response data) images of skin patches to detect the areas of skin lesion. The Response data comprises of binary mask images in PNG format which indicates the location of the primary skin lesion within each input lesion image.
![Objective](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Objective.PNG)
***

# Data Loading:

- An Implementation of the User Input is developed to provide the relevant path for the Raw data set (Explanatory data) and Segmented dataset (Response data).
- Using the OS library all the images from the above-mentioned path were stacked/stored. 
- Tqdm library is called in order to confirm and check the status of storing the entire dataset in kernel.
- Images were sorted in an ascending order with respect to labels to maintain the correct sequence and consistency of all the images.

# Data Pre-Processing:

- The Images were resized to `(256 x 256-pixel)` values to maintain the uniformity throughout the modelling.
- Splitting of explanatory and response variable was implemented into Training, Validation, and Testing set with `60:20:20` ratio respectively. The images are resized to `(256x256)`and the colour made used for the representation of dataset is greyscale. 
- Further, Normalization of Explanatory variable is done by dividing the dataset by 255 and the response variables are rounded off to 0 or 1 as they  are encoded in single channel (grayscale) with 8-bit PNGs where each pixel is either of 0 (background of the image, or areas outside the primary lesion) or 255 (the foreground of the image, or areas inside the primary lesion)
- Finally, One-Hot encoding was applied into the response variable which will offer more nuanced way of prediction.

# Model Architecture:
![Model_Architecture](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Model_Architecture.PNG)
The **U-Net** architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions.
Here, the implementation of Improved U-Net is developed by referring the article published by F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein [1]

- **The Contraction Path:** This is composed of 4 blocks, with each block consist the following architecture:
	- Context Module which is the pre-activation residual block is implemented in each of the 4 blocks, which consist of two convolutional layers of `(3x3)` kernel size and in between which there is also a dropout layer of 0.3 to avoid overfitting of the model.
	- Between two Context Module, a convolutional layer of `(3x3)` is added with a stride of 2 to reduce the resolution of the feature maps and will allow for more features while descending the aggregation pathway.

- **The Expanding path:** This is composed of 4 blocks, with each block consist the following architecture:
	- Upsampling2D layers have been used in each of the 4 blocks to increase the image dimensions. 
	- Each Contraction Path is concatenated with the output of down sampling part, to produce the instance result from the model. 
	- Localization module is constructed which consists of a `(3x3)` convolutional layer and a `(1x1)` convolutional layer which are combined via elementwise summation to form the final network output.
	- Finally, different segmented layers have been merged using element wise summation to administer results more precisely.

The segmentation performance of participating algorithms is measured based on the **DICE coefficient**, where DICE coefficient is measured as:
`DSC=  (2|X∩Y|)/(|X|+|Y|)`

Activation function used is _**“Leaky ReLU"**_ nonlinearities with a negative slope of `〖10 〗^(-2)` for all feature map computing convolutions.  The loss function used is _**Binary_Crossentropy**_ with _**Sigmoid**_ as the Output layer Activation function as the prediction of segmentation comprises of binary classification.

# Model Output:

On the ISIC training data set, the built U-Net model is trained and the best model selection takes place based on validation data set loss improvement. The call-back parameter was allocated to catch the best model that had the lowest loss of validity.
- **Binary_Crossentropy Loss plot** with respect to the number of epochs for training and validation data set along with marker for best model is represented below.

![Binary_Crossentropy_Loss Plot](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Model_Loss_Plot.png)

- **Accuracy Plot** with respect to the number of epochs for training and validation data set along with marker for best model is represented below.

![Accuracy Plot](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Model_Accuracy_Plot.png)

- Overall **Dice Score** of the ISIC test data set obtained is `0.85`.
`tf.Tensor(0.85157216, shape=(), dtype=float32)`
Further, it was also idetified and confirmed  that all the test dataset is having the dice score of above `0.80`.

- **Predicted segmentation** image of a random ISIC test sample with lesion image and  actual segmentation image is represented below.

![Predicted_Image1](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Predicted_Image1.png)
![Predicted_Image2](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Predicted_Image2.png)
![Predicted_Image3](https://github.com/Raghav-Dhanuka/PatternFlow/blob/topic-recognition/recognition/Raghav-Dhanuka/Predicted_Image3.png)

***
# Dependencies:

- There are few packages required to be installed and imported before running the algorithm such as _OS, Numpy, Pandas, Matplotlib, Keras, Scikit-Learn, Scikit-Image, tqdm_.
- The algorithm script name - _**“Improved_Unet_ISICs_Algo.py”**_ and the Main test drive script name - _**“main. py”**_ should be kept in the same path before the execution of the test drive script.
- The output of the algorithm is present in the _**Output_Model.ipynb**_ file which depicts the plots and images of the model result.

# References:

[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and
Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
