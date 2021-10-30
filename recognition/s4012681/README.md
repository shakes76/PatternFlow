# Classification of MR images of the male pelvis using U-net3D

The 3D Medical Radiation Images (MRIs) being classified in this model 
have been made available by the CSIRO and were collected as part of a radiation 
therapy study from the Calvary Mater Newcastle Hospital, available at 
[https://data.csiro.au/collection/csiro:51392v2](https://data.csiro.au/collection/csiro:51392v2).  
The collection also contains the segmentation masks for the MRIs. 

The goal of this model is to segment the MRIs in order to classify the body, bones, 
bladder, rectum and prostate.

The dataset consists of 211 MRIs and masks from 42 different patients with weekly 
images taken over up to 8 weeks.  Using a random number generator the images were 
split up by patient to give a split of approximately 70% of the images for training, 
20% for validation and 10% for testing. This resulted in the split:

| Train Case Numbers | Validation Case Numbers | Test Case Numbers |
|--------------------|-------------------------|-------------------|
|Case 4 | Case 17 | Case 5|
|Case 6 | Case 20| Case 11 |
|Case 8 | Case 23 | Case 25 |
|Case 9 | Case 29 | |
|Case 10 | Case 33 | |
|Case 12 | | |
|Case 13 | | |
|Case 14 | | |
|Case 15 | | |
|Case 16 | | |
|Case 18 | | |
|Case 21 | | |
|Case 22 | | |
|Case 24 | | |
|Case 26 | | |
|Case 27 | | |
|Case 28 | | |
|Case 30 | | |
|Case 31 | | |
|Case 32 | | |
|Case 35 | | |
|Case 36 | | |
|Case 37 | | |
|Case 38 | | |
|Case 39 | | |
|Case 40 | | |
|Case 41 | | |
|Case 42 | | |


The architecture of this U-net3D model was taken from Çiçek et al. “3D U-Net: 
Learning Dense Volumetric Segmentation from Sparse Annotation”.

![U-net3D Architecture](unet_architecture.png)

In the contraction section each layer consists of a 3 x 3 x 3 3D convolution with 
ReLu activation, eac followed by batch normalisation and finally a max pooling of 
2 x 2 x 2 with strides of size 2.  
In the expansion section each layer consists of a Conv3DTranspose of 2 x 2 x 2 
with strides of size 2 followed by a concatenation with the equivalent contraction 
layer.  Then two convoultions of 3 x 3 x 3 with ReLu activation and batch 
normalisation.
Finally a convolution of 1 x 1 x 1 is applied with softmax acitavtion reduces the 
number of channels to six, one for each of the classifications.

|Accuracy|Loss|
|---|---|
![Accuracy and Loss](acc_loss.png)

