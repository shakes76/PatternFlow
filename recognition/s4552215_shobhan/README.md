**Submitted By: Shobhan Mitra**

**Student ID:45522156**

**The Objective:**

To Segment the ISICs data set with the UNET, in accordance that the predicted segmentation response with the test data set should have a minimum dice co-efficient of 0.7.

Where, Dice Score = (2\*Area of Overlap between the Images)/(Total number of pixels on both images)

**Data Set:**

Input – The input data set are dermoscopic lesion images in JPEG format.

Response Data –The Mask images are PNGs of grayscale, where each pixel is either 0(Background of image) or 255(foreground of image).

**Train, Validation and Test Split:**

The overall sample size is 2594 and the train, validation and test split ratio is 60:20:20. (1556,519,519)

**UNET Architecture:**

UNET is a convolutional Neural Network. One of the promising feature of UNET is it can achieve more precise segmentation with fewer training images, which made is very popular in Biomedical Image segmentation.

The network architecture of UNET consists of two main parts. First path is the contraction path(encoder), comprised of convolutional and max pooling layers to capture the context of image. The second part is expanding path(decoder), comprised of transposed convolutional and upsampling network to enable precise localization.One of the unique features of U-NET is that it uses skip connections between the contraction and expansion path which performs a concatenation operator instead of a sum. These skip connections help in transferring local information to the global information while upsampling.

The U-NET architecture is shown below: -

![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/UNET_architecture.png)

In a nutshell, we have the following relationship: Input (128x128x1) =\&gt; Encoder =\&gt;(8x8x256) =\&gt; Decoder =\&gt;Output (128x128x1).

**Model Training and Best Model Selection**

The UNET model is trained on the ISIC training data set and the best model has been selected on the basis of improvement of validation data set dice loss. To capture the best model with the lowest validation dice loss, call back parameter is used.

**Required Python Files:**

1. **Data\_Load\_ISIC\_SM:** Contains function to upload lesion and segmentation images of the ISIC data set.
2. **Image\_Segment\_Preprocess\_ISIC\_SM:** Contains function to preprocess data which includes hot encoding to split into two categories such as background or foreground images.
3. **UNET\_Model\_ISIC\_SM:** Contains function to build UNET model.
4. **UNET\_Model\_Compile\_ISIC\_SM:** Compile the model with allhyperparameters such as optimizer,loss,accuracy etc. This also evaluate dice scores along with training.
5. **Driver\_Script\_ISIC\_SM:** To run the total code through command line prompt.

**Running the code in command line with Driver script:**

To run the code in command line we will follow the below steps:

1.Create environment

2.Allocating path to data.

3.All files should be in same path.

![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Driving_Script_Command.PNG)

**Training Model:**

I have attached few snippets of network architecture and converging stages from model training session as follows:

![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Model_Archi.png)
![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Converging_and_Output.PNG)

**Output:**

- Dice Loss with respect to the number of epochs (Samples shown below).
![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Figure_1_Dice_Loss.png)

- Accuracy plot with respect to the number of epochs (Samples attached below).

![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Figure_2_Classification_Accuracy.png)

- Dice scores as CSV file for all predicted segmentation images (saves in the selected path during training), attached in repository as Dice\_Coefficients\_Test.csv.
- A random ISIC test sample with lesion image, actual or ground truth segmentation image and predicted segmentation image (Sample attached below).
![](https://github.com/smitraDA/PatternFlow/blob/topic-recognition/recognition/s4552215_shobhan/Figure_3_Segmentation_output.png)

- **Conclusion:**

UNet is very popular and strong enough to do image localisation by predicting the image pixel by pixel. There are many applications of image segmentation especially in Biomedical using UNet.