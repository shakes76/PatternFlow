# Project description:
Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

## Dataset description:
The data set contains and the original skin patch image segmentation, used to detect melanoma (skin cancer) area. The depth of neural network in this subset of a data set is the purpose of the training, then split test image, in order to identify areas have melanoma. The images in the dataset are of different sizes and then modified to fit the model in the pre-processing stage.

## Unet architecture
<image width="700" src="Screenshots/Unet.png" />

## Improved Unet architecture
<image width="700" src="Screenshots/Improved unet.png" />
