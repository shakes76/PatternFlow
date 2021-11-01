# Using the Improved U-Net to perform image segmentation on the ISIC dataset


U-Net image segmentation is an important architecture to be utilised when needing to generate image masks for a variety of classes. In this program, we utilise an improved version  of the U-Net architecture that was proposed in the paper "Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge," **by F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein.* The predominant change between the standard U-Net architecture and the one they proposed, is the addition of context modules that utilises a LeakyReLU layer, two additional convolutional layers and a drop out layer of 0.3. 

[Improved U-Net Architecture]("./Imp Unet Architecture.JPG")


Through this, the convolution sizes are dramatically reduced, with U-Net normally beginning with feature maps that start at 64 filters  and then grow in size to 1024 filters in the last layers. Improved U-net begins at 16 filters and only grows to 256 filters before being upscaled on each layer. 

The ISIC dataset, is a dataset comprised of 2,594 images of various skin lesions and their relevant segmented masks. There is a lot of variety in the look, shape and size of the lesions, one example of them is the below

[ISIC - Lesion]("ISIC - Lesion.JPG") [ISIC - Mask]("ISIC - Mask.jpg")

By utilising the Improved U-net architecture, I was able to generate masks that across the Test Dataset, have a Dice accuracy (Calculated by (2TP/(2TP + FP + FN)) of approximately 83-85%. This is an example of a mask generated for the above photo. 

[Mask Generated via Improved U-Net]("Unet - Mask.JPG")

