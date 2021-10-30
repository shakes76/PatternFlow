# Classification of MR images of the male pelvis using U-net3D
---
## The data
The 3D Medical Radiation Images (MRIs) being classified in this model 
have been made available by the CSIRO and were collected as part of a radiation 
therapy study from the Calvary Mater Newcastle Hospital, available at 
[https://data.csiro.au/collection/csiro:51392v2](https://data.csiro.au/collection/csiro:51392v2).
The collection also contains the segmentation masks for the MRIs. 

The goal of this model is to segment the MRIs in order to classify the body, bones, 
bladder, rectum and prostate.

The dataset consists of 211 MRIs and masks from 42 different patients (labelled 
as cases) with weekly images taken over up to 8 weeks.  To try and prevent data 
leakage a random number generator was used to split the dataset by case, so 
the same patient would not appear in more than one set.  The samples were 
split to use approximately 70% of the images for training, 
20% for validation and 10% for testing. This resulted in the split:

| Validation Case Sample | Test Case Sample | Train Case Sample |
|------------------------|------------------|-------------------|
| Case 17 | Case 5| Remaining Cases
| Case 20| Case 11 |
| Case 23 | Case 25 |
| Case 29 | |
| Case 33 | |

## The model

The U-net is a fully convolutional model which has an encoder path using 
pooling to contract and a decoder path using transpose to upsample the image. 
This is combined with skip connections between the layers in each path which 
help to preserve finer detail which may otherwise be lost.

The architecture of this U-net3D model was based on Çiçek et al. “3D U-Net: 
Learning Dense Volumetric Segmentation from Sparse Annotation”:

![U-net3D Architecture](unet_architecture.png)

In the encoder section each layer consists of two 3 x 3 x 3 3D convolutions 
with ReLu activation, with a single batch normalisation between and finally a 
max pooling of 2 x 2 x 2 with strides of size 2.  
In the decoder section each layer consists of a Conv3DTranspose of 2 x 2 x 2 
with strides of size 2 followed by a concatenation with the equivalent encoder 
layer.  Then two convolutions of 3 x 3 x 3 with ReLu activation with a batch 
normalisation between.
Finally, a convolution of 1 x 1 x 1 is applied with softmax activation reduces 
the number of channels to six, one for each of the classifications.

## Performance

Over 50 epochs the model consistently obtains a Dice similarity coefficient 
for each category of at least 0.70 on the test set.

|Accuracy|Loss|
|---|---|
![Accuracy and Loss](acc_loss.png)

