# Title
Classifying ISIC dataset with Unet with dice coefficient as evaluation metric

*Submitted by- Tushar Vinayaraj, s4608425*

## Description of algorithm
The model used is Unet, what happens here is, there are three stages, In the first stage, the first four levels are the image contracting layers where the size of the image is reduced, done by using, conv2d layers of tensorflow's keras library along with maxpooling2d layers. Then the second stage is bottleneck stage which is again done by conv2d layers. The third stage is the expansion stage where the images regain their original size along with segmentation, this is done by four layers of conv2d and upsampling2d. Padding is also done in the model as a precaution that the image arrays do not loose original information while reduction.

[!unet](https://github.com/tusharvinayaraj/PatternFlow/blob/topic-recognition/recognition/s4608425/resources/unet.PNG)


### Problem it solves
This deep learning model helps to solve the problem of segmentation of the images that can be used for various purposes in the medical and healthcare industry.The output of this deep network is compared with the original input using a metric known as dice similarity. It tells us how well the image segmentation resembles the original input image based on the overlapping. It is calculated by twice the area overlapping between segmented and original image upon the total pixels in segmented and original image. 

### How it works
At beginning, all the images of the ISIC dataset were sized same into 196,256 as it was all differently sized. The 255 segmented classes were converted to binary classes by normalisation (/255) and then by approximating labels below a mid-threshold into one class and above threshold into another class. The filtered images were passed in the model that is described above. The output images came out to be segmented(they only had two colours, one for each class) and hence, the goal of the problem was achieved. The train test validation split is at a ratio of 70:30:30.

### Result
This model achieved a dice similarity coeffient of approximately 0.62.

### Figures
The loss plot
[!loss](resources/loss.PNG)

example output
[!op](resources/output.PNG)


### Bibliography
1. Ekin Tuiu, "Metrics to evaluate your semantic segmentation model", https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=Simply%20put%2C%20the%20Dice%20Coefficient,of%20pixels%20in%20both%20images.

2. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in
Computer Science, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241


