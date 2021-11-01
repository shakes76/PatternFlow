# Segment the ISICs data set with the Improved UNet
Author: Tuo Yuan (46095312) 

### Usage
firstly, run the `preprocessing.py` to load the data from ISIC dataset, 
it splite the data to

- data_train
- data_test
- data_val
- mask_train
- mask_test
- mask_val

as .npy file.

run `train.py`[2] to start trainning the model which is in the `model.py`[3]

finally, run `driver.py`[2] to evaluate/test the performance

### Discription
This module uses an improved U-Net neural network with 
all labels having a minimum DIce similarity coefficient 
of 0.8 for image segmenetation on the ISICs 2018 dataset.


### Network architecture
![Improved_UNet](output/Improved_UNet.jpg "Improved_UNet")

this method make use of the U-Net network archietecture[1]. 
The context pathway aggregates hih level information that is 
subsequently localized precisely in the localization pathway.

### Challenge
[1]In medical image segmentation problem, due to the class imbalance in the datasets, Dice coefficient performs better at class imbalanced problems than cross-entropy, it measures the similarity between two sets of data, in this case, it compare pixel to pixel between mask labels and train_data.

##Results

### output performance
- **Area under the ROC curve**: 0.8567094851762822
- **Area under Precision-Recall curve**: 0.8925321138822117
- **Jaccard similarity score**: 0.7493366507185417
- **F1 score (F-measure)**: 0.8567094851762821
  
![test results](output/sample_results.png "Sample results")
![test results](output/Precision_recall.png "Precision_recall")
![test results](output/ROC.png "ROC")
![loss](output/loss.png "loss")
![accuracy](output/accuracy.png "accuracy")

###references:
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
Available: https://arxiv.org/abs/1802.10508v1

[2] https://github.com/languede/BCDU-Net

[3] https://github.com/shakes76/PatternFlow