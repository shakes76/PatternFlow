
Title: Image Segmentation (Improved UNet on ISICs data set) by Xiao Sun

Student name: Xiao Sun;
Student ID: 45642586;


For COMP3710 Pattern Recognition Report, I choose the fourth problem, which is: 
4. Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity coefficient of 0.8 on the test set. [Normal Difficulty]

Description of algorithm and the problem solved:
Our model algorithm is an improved UNet model, which is inspired by the popular U-Net model.

The data set we used is part of ISIC 2018 challenge data for skin cancer segmentation labels (preprocessed version).
In this project, we aim to implement the improved U-Net model and apply on ISIC data set to segment skin cancer images.

......describe and justify your training, validation and testing split of the data
......problems(overfitting), how i change algorithm, and why 

Usage of the module (how it works):
Please run the driver script (driver.py) to call the Improved UNet module in UNet_model.py. The UNet_model is imported by driver.py 
and train/test on ISICs 2018 data set to finish the image sementation, where the Improved UNet module is implemented based on Tensorflow.
In the driver script, we import ISICs 2018 data set and also import the Improved UNet module from UNet_model.py. If we would like to 
apply the Improved UNet module on other dataset, the UNet_model.py can be directly imported, then use the Improved_UNet_model function and 
modify the parameters and hyperparameters.


Visualisation:



Output plot (example of input/ground_truth/predicted_mask):


Reference:
F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and
Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available:
https://arxiv.org/abs/1802.10508v1
