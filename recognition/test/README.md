"# test" 

==============================================================================================
#Pattern Recognition

===============================================================================================
---Segment the OASIS brain data set with an Improved UNet with all labels having a minimum Dice
   similarity coefficient of 0.9 on the test set.
--------------------------------------------------------------------------------------------------

##Description

This report illustrates how to train a U-Net for OASIS brain tumor segmentation, using a minimum   
Dice similarity coefficient of 0.9 on the test dataset. The data is consist of MR images  
of the 300+ human brains with accompanying segmentations, which we need to download from Cloidstore   
with the link 'https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download'.

##Environment

*tensorflow version: 2.3.0 
*python version: 3.7.9


##Alogrithm   
* Unet_Model:
	1.Parameters:

*Parameters:  



solves and how it works
The Unet_model
Depending on the U-Net architecure, we designed the network to process large 2D input block of 256 * 256 voxsels.   



visualization