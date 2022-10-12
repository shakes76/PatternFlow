# Super-resolution network

## SRCNN Brief with the problem
SRCNN is a deep learning method (super rolution CNN) for reconstructing original image with high-resolution. I implemented SRCNN which takes down-sampled MRI images with factors of 4 (low-resolution images) and produce a "reasonably clear image of original size.

### SRCNN theory and structure
####  SCRNN theory
SCRNN using a deep CNN to directly learn an end-to-end mapping between the low-resolution images and the high-resolution images. The actual CNN model is not complex and deep comparing with other more advance model as it is one of the initial proposed model using cnn to do the image reconstruction. It contains three layers that each handles a different task which will be shown in the structure part below. Basically, it takes a resized image and upscale to the original size using bicubic interpolation, then feed into the model of SRCNN. Since upscaled-image is indistinct but the size is same as original, SRCNN will conduct patch extraction from this image and using non-linear mapping，finally reconstruct the high-resolution image as output 
#### SCRNN Structure	
![SRCNN structure](/SRCNN_model_structure.png)
**Layer1:64, (9,9)**
**Layer2:32, (1,1)**
**Layer3:3,(5,5)**

## 2. Setting up
### Environment Guide
1. open up ananconda prompt shell and create tensorflow environment with version **2.3.0**, check tensorflow version as shown below
![tensorflow environment](/environment.png)
and make sure downloaded tensorflow with GPU version and check GPU existed
2. using "pip install matplotlib" 
the matplotlib verison should be **3.5.2**
and "pip install xxx" to download other missing package
or "conda install XXX" if appropriate

#### Alternative option 
which is how i did for training and testing the model, use the .ipynb run the code in colab to train the model and save the model to your local computer and run the predict.py to see the output image result. 
Rember to change the appropriate path 
## 3. Example inputs, outputs and plot of algorithm
**Input**
![down-scaled](/predict_result/down-scale.png)
UpScaled to the same size as output to feed the model
![resized by tensorflow function](/upscale.png)

**Output**
![image_comparion](/predict_result/result_comparison.png)
The first one is original image and the second one is down-scaled of by 4, the third one is resized back to original size by tensorflow function, the last one is reconstruct by SRCNN model. Since plt auto size the image, therefore the downscale image cannot see the size is different i took screenshot from running the the code in ipynb cell, you can view all the all four comparison image in predict_result folder

#### original
![original](/predict_result/original.png)
#### down-scaled
![down-scaled](/predict_result/down-scale.png)
#### resized by tensorflow function
![resized by tensorflow function](/predict_result/upscale.png)
#### SRCNN 
![SRCNN](/predict_result/SRCNN.png)
From paper as iterations increases, the output image will become more clear. Due to GPU limitation, this is my current model output

## 5. Pre-processing
download the dataset from bb course help page and unzipp file
![loss and accuracy metric plot](/plot.png)