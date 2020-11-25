"Author" 
Zhifang Zheng(45788167)

Pattern Recognition
==============================================================================================

Basic Description 
--------------------------------------------------------------------------------------------------
This report illustrates how to train a U-Net for OASIS brain tumor segmentation, using a minimum   
Dice similarity coefficient of 0.9 on the test dataset. The data is consist of MR images  
of the 300+ human brains with accompanying segmentations, which we need to download from Cloidstore   
with the link 'https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download'.  

There are 2 scipt files, a README.md file, and one image file( final results and model_summary)

Environment
--------------------------------------------------------------------------------------------------
*tensorflow version: 2.3.0    
*python version: 3.7.9

Alogrithm  
--------------------------------------------------------------------------------------------------
* Unet_Model: Depending on the U-Net architecure, we designed the network to process large 2D input block of 256 * 256 voxsels.  
	*Parameters:
	- out_channel
	- input_data
	- activation
	- epoches  
	- train_ds.batch  
	- val_ds.batch

   *Loss Function:  
	- dice_coef: metrics 
	- dice_coef_loss  
--------------------------------------------------------------------------------------------------
The final test result will show in ReadMe.md. Pleast reference it to test whether this Unet_model is good.
