
# Title: Image Segmentation (Improved UNet on ISICs data set)

## Author
Student name: Xiao Sun;

Student ID: 45642586;

## Question
For COMP3710 Pattern Recognition Report, I choose the fourth problem, which is: 
4. Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity coefficient of 0.8 on the test set. [Normal Difficulty]


## Description of algorithm and the problem solved:
Our model algorithm is an improved UNet model (U-shaped encoder-decoder structure), which is inspired by the popular U-Net model.

![image](https://user-images.githubusercontent.com/69885082/98076019-2eed0380-1eb9-11eb-9976-0f9daab0286b.png)


The data set we used is part of ISIC 2018 challenge data for skin cancer segmentation labels (preprocessed version).
In this project, we aim to implement the improved U-Net model and apply on ISIC data set to segment skin cancer images, to recognition the skin cancer area from skin images.

I split the whole data set into training/validation/testing set (70:15:15).

Our first version algorithm has overfitting issue, the training DSC is very hight but validation/test DSC are less than 0.75. Therefore, because our U-Net model is very deep, so I add some batch_nomalization layers after some of the con2d layers to solve overfitting. Besides, I also set the batch size as 32.


## Dependencies:
The following are required:

	1. Python 3.7
	
	2. Tensorflow-gpu 2.1
	
	3. Matplotlib
	
My environment.yml file can be used to create env for ease of use with Anaconda.

## Usage of the module (how it works):
Please run the driver script (driver.py) to call the Improved UNet module in UNet_model.py. The UNet_model is imported by driver.py 
and train/test on ISICs 2018 data set to finish the image sementation, where the Improved UNet module is implemented based on Tensorflow.
In the driver script, we import ISICs 2018 data set and also import the Improved UNet module from UNet_model.py. The Improved UNet module can be re-used on other dataset by import UNet_model.py, then use the Improved_UNet_model function and parameters/hyperparameters can be modified. The training parameters can be modified in driver.py.
## Algorithm
  UNet_model.Improved_UNet_model()
* __Parameters:__
	1. __filters  : int__ 
	
		filters of the first conv2d layer. By default is 16.
				
	2. __input_layer  : tf.keras.Input tensor__ 
  
		tf.keras.Input layer, is used to instantiate a Keras tensor. By default is tf.keras.Input((256,256,3)).

* __Returns:__

	1. __improved_unet_model : tf.keras.Model__
 
		The U-Net model. tf.keras.Model groups layers into an object with training and inference features.

## Visualisation:
Below is the plot of train/val Dice similarity coefficient of 200 epochs:

![image](https://user-images.githubusercontent.com/69885082/98068592-30fa9680-1ea8-11eb-800f-9520fbfd2390.png)

The training DSC reaches 0.95 and validation DSC is around 0.82 after 200 epoches.
Test DSC is shown as below, which is 0.805:

![image](https://user-images.githubusercontent.com/69885082/98068749-a6fefd80-1ea8-11eb-8dc2-b7b59676014d.png)

## Output plot (example of input/ground_truth/predicted_mask):

![image](https://user-images.githubusercontent.com/69885082/98069299-293bf180-1eaa-11eb-8202-58844d1e9a9b.png)
![image](https://user-images.githubusercontent.com/69885082/98069323-335df000-1eaa-11eb-8ab1-a48ff7d219a7.png)
![image](https://user-images.githubusercontent.com/69885082/98069340-3eb11b80-1eaa-11eb-99ff-d689dbcd1b1e.png)
![image](https://user-images.githubusercontent.com/69885082/98069331-38bb3a80-1eaa-11eb-9a8f-b650de0de17c.png)
![image](https://user-images.githubusercontent.com/69885082/98069333-3bb62b00-1eaa-11eb-99cf-265343c8ac4a.png)
![image](https://user-images.githubusercontent.com/69885082/98069352-47095680-1eaa-11eb-8abb-2378b7345e2c.png)
![image](https://user-images.githubusercontent.com/69885082/98069371-52f51880-1eaa-11eb-970f-a6d45996c27c.png)
![image](https://user-images.githubusercontent.com/69885082/98069377-56889f80-1eaa-11eb-807a-ba709763704a.png)
![image](https://user-images.githubusercontent.com/69885082/98069382-58eaf980-1eaa-11eb-8277-f3be398eb0b0.png)
![image](https://user-images.githubusercontent.com/69885082/98069385-5ab4bd00-1eaa-11eb-8337-6ba1d7ed3d32.png)




Reference:
F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and
Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available:
https://arxiv.org/abs/1802.10508v1
