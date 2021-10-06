<h2>
Segment the ISICs data set with the Improved UNet 
</h2>
<b>Created by Anqi Yan, S4413599</b>
<p>
This project used Improved Unet to segment the ISICs dataset into foreground and
background segmentations. Aim to achieve above 80% of Dice Similarity Coefficient.<br><br>

The ISICs data contains dermatology images and its segmentation images. The Improved UNet takes dermatology images as input, takes its segmentation images as output to train the Improved UNet Model.
</p>

<h3>Improved UNet</h3>
The project used Improved UNet model which is derived from the U-Net to do the segmentation.
<p align = "center">
    <img src = "./IMG_Folder/improved_unet.png" />
</p>
The Improved UNet diagram shows above. <br>
The CONTEXT module is a pre-activation residual block
which contains 3*3*3 convolutional layers and a dropout layer with 
0.3 drop rate<br>
The upsampling module contains a upsampling2D layer and a 3*3*3 convolution
layer<br>
The Localization module contains a 3*3*3 convolutional and a 1*1*1
convolution layer<br>


<p2>
Running result shows below
</p2>
<p align="center"> 
	<img src="./IMG_Folder/history_dsc.png" />
	<img src="./IMG_Folder/loss_valoss.png" />
	<img src="./IMG_Folder/val_train_acc.png" />
</p>