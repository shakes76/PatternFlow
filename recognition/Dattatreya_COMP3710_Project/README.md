**Objective**

Applying U-NET architecture on ISIC data set to predict segmentation responses on lesion images in test data set with an overall minimum Dice Score of 0.7.

**Data Set**

**Input**  **–** The input data set consists of dermascopic lesion images in JPEG format. The lesion images were extracted from a variety of dermatoscope types, from all anatomic sites, historical sample of patients presented for skin cancer screening and several institutions.

**Response Data** – It comprises of binary mask images in PNG format which indicates the location of the primary skin lesion within each input lesion image. Mask images are encoded as single channel (grayscale) 8-bit PNGs where each pixel is either of the two cases mentioned below: -

1. 0: Which represents the background of the image, or areas outside the primary lesion
2. 255: Which represents the foreground of the image, or areas inside the primary lesion

**Goal**

The predicted segmentation responses based on the U-NET model on the ISIC test data set should have an overall minimum dice score of 0.7. Dice score can be mathematically expressed using the following formula in case of images: -

_Dice Score = (2\*Area of Overlap between the Images)/(Total number of pixels on both images)_

**U-NET Architecture**

The U-Net is a convolutional network architecture designed for fast and precise segmentation of images. The algorithm has outperformed the prior best method of sliding-window convolutional network and won various international challenges. The network architecture is U-shaped which is symmetrically designed and consists of two major parts – the left part is called the contracting path which comprises of the general convolutional processes while the right part is expansive path which is constituted by transposed 2d convolutional layers. One of the major features of U-NET is that it comprises of skip connections between the contraction and expansion path which applies a concatenation operator instead of a sum. These skip connections assist in transferring local information to the global information while upsampling. The U-NET architecture is shown below: -

![UNET Image](https://github.com/Dattatreya-45586327/PatternFlow/blob/topic-recognition/recognition/Dattatreya_COMP3710_Project/Images/UNET.png)

**Contracting Path Components**

The contracting path is composed of 4 blocks. Each of these blocks is composed of: -

- 3x3 Convolution Layer + activation function (with batch normalization)
- 3x3 Convolution Layer + activation function (with batch normalization)
- 2x2 Max Pooling

**Bottleneck**

It is located between the contracting and expanding paths. It is built from 2 convolutional layers with batch normalisation and dropout

**Expanding Path Components**

It also composed of 4 blocks and each block comprises of: -

- Deconvolution layer with stride 2
- Concatenation with the corresponding cropped feature map from the contracting path
- 3x3 Convolution Layer + activation function (with batch normalization)
- 3x3 Convolution Layer + activation function (with batch normalization)

**Train, Validation and Test Split**

For this problem the train, validation and test split was in the ratio of 60:20:20. As the overall sample size was 2594 so a higher weightage was given on the overall training and validation set, so that the model can be robust

**Model Training and Best Model Selection**

The developed UNET model is trained on the ISIC training data set and the best model selection takes place on the basis of improvement of validation data set dice loss. Callback parameter has been assigned to capture the best model which has the lowest validation dice loss.

**Documentation of Python Code Files**

- **ISIC\_Data\_Load.py** – This file has functions pertaining to uploading lesion and segmentation images of the ISIC data set.
- **Img\_Seg\_Preprocess.py** – This file pertains to image pre-processing of both the lesion and segmentation images. The segmentation images are also converted into categorical data for the based on the nature of the individual pixels i.e. whether they are background or foreground images.
- **UNET\_Model.py** – This file contains functions for building the UNET architecture by constructing the contraction and expansion parts and using the proper activation and batch normalisation techniques.
- **UNET\_Mod\_Compile.py** – This file encapsulates functions for evaluating overall and segmentation image wise Dice Scores, compiling the model by assigning appropriate optimizers, loss functions and accuracy metrics. Callback parameter has also been initialised to implement early stopping and storing the best model based on the performance on the validation data set.
- **Driver\_Script.py** – This file can be executed from the command terminal by passing the lesion image and segmentation image paths. This script executes the entire code and evaluates the model based on Dice Score.

**Usage of the Files**

- Make sure that all the above-mentioned files are kept in the same path
- The execution should be carried out in the environment where all the packages are installed
- The driver script needs to be executed in the following way in the command terminal after setting up the environment: -


**python Driver\_Script.py Input\_Image\_Path Response\_Segmentation\_Path**

**But make sure you add the &quot;/&quot; after each of the paths for the code to execute**

The code execution command line snippet is provided below for reference: -

![Code Execution](https://github.com/Dattatreya-45586327/PatternFlow/blob/topic-recognition/recognition/Dattatreya_COMP3710_Project/Images/Code_Exec.PNG)

**List of Packages Required**

- os (pre-installed)
- keras
- tensorflow
- numpy
- pandas
- itertools
- tqdm
- scikit-image
- matplotlib
- random
- sys

**Expected Output**

- Dice Loss and Accuracy Plots with respect to the number of epochs for training and validation data set along with marker for best model(Samples shown below)

![Dice Loss](https://github.com/Dattatreya-45586327/PatternFlow/blob/topic-recognition/recognition/Dattatreya_COMP3710_Project/Images/Dice_Loss_Plots.png)

![Accuracy](https://github.com/Dattatreya-45586327/PatternFlow/blob/topic-recognition/recognition/Dattatreya_COMP3710_Project/Images/Accuracy_Plots.png)

- Overall Dice Score of the ISIC test data set
- CSV file with individual dice scores for all predicted segmentation images for the test data set (will be saved in the path where the code is executed, Sample Provided)
- Plot of a random ISIC test sample with lesion image, actual segmentation image and predicted segmentation image (Sample shown below)

![Predicted Segmentation](https://github.com/Dattatreya-45586327/PatternFlow/blob/topic-recognition/recognition/Dattatreya_COMP3710_Project/Images/Sample_ISIC_Predicted_Plot.png)