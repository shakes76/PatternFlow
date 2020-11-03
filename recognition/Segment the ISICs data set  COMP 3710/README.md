# Improved Uneter
## 1. Requirements   
```bash
tensorflow>=2.2
scikit-learn
numpy
cv2
``` 

## 2. Algorithm Description   
The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).  
The model is used to automatically predict the segmentation boundary of skin lesions.
### Model
 ![picture/u-net-architecture.png](picture/u-net-architecture.png)
 The above is the original model structure proposed in the paper. On the basis of the original model, I modified the four copy and crop and added a 2x2 convolution layer. Using ISIC2016 data set for training, dice loss is used as the loss function, and dice similarity coefficient is used as the evaluation standard. The score of 0.88 is achieved in ISIC16 test set.

### Result
Here are the test results
From left to right are test.jpg pre.jpg mask.jpg
 ![picture/result.png](picture/result.png)
Here's the visualization.
![recognition/Segment the ISICs data set  COMP 3710/acc.png](recognition/Segment the ISICs data set  COMP 3710/acc.png)
![picture/dice.png](recognition/Segment the ISICs data set  COMP 3710/dice.png)
![picture/loss.png](recognition/Segment the ISICs data set  COMP 3710/loss.png)
![picture/lr.png](recognition/Segment the ISICs data set  COMP 3710/lr.png)
 



## 3. Train
1. Download the dataset and unzip it under datasets
```
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip
```
2. Install the libraries necessary to run the code
3. 
```
python train.py --batch_size 16 --data_dir datasets --workers 8 --epochs 100 --lr 0.0001 --logs ./logs 
```
## 4. test
Test model performance on isic2016 and Dice similarity coefficient
```
python test.py --data_dir datasets --model weight/best.ckpt
```
If you don’t want to train the model from scratch, you can download my trained model from [here] (fill in the model link)

If you want to see the model effect more intuitively
```
python pre.py --pre_dir test --model weight/best.ckpt --out testout
```
This will input the images in the test into the model, and convert the output into images and store them in the testout
