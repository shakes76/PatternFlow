# Title
Classifying ISIC dataset with Unet with dice coefficient as evaluation metric

## Description of algorithm
The model used is Unet...

### Problem it solves
This deep learning model helps to solve the problem of segmentation of the images that can be used for various purposes in the medical and healthcare industry.


#### How it works
At first, all the images of the ISIC dataset were sized same into 196,256 as it was all differently sized. The 255 segmented classes were converted to binary classes by normalisation (/255) and then by approximating labels below a mid-threshold into one class and above threshold into another class. The filtered images were passed in the model that is described above. The output images came out to be segmented(they only had two colours, one for each class).

#### Bibliography


