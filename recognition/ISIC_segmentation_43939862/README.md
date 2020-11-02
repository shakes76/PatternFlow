# ISIC Image Segmentation Using Improved Unet

Improved Unet[1] is a ConvNet which is adapted from the original Unet[2], with a few added features to improve performance. 
The two main additions in Improved Unet is the use of residual connections in the encoder, and the use of segmentation layers in the decoder. 
The residual connections[3] help with gradient-vanishing by having activations skip layers and be added element-wise to the activations of the skipped layer. 
Segmentation layers[4] allow information from throughout the decoder to be used in the output layer directly, with the goal of improving the quality of the feature maps in the layers from which the segmentation layers are generated.
In this project, Improved Unet is applied to the ISIC 2018 (task 2) dataset, with the goal of segmenting the skin cancer images with an average DICE score of at least 80% on the test set. 
This dataset has been pre-processed, and includes 2596 RGB images of varying resolutions, along with their ground truth segmentations.

## The Algorithm
### Pre-processing and model-handling
**File: driver.py**
1. Image/label filenames are read and split into training/validation/test sets.
2. Sets are formated into tensorflow datasets and shuffled.
3. Map the datasets according to a pre-processing function, allowing pre-processing to be done "on the fly".
	a) Load/decode image file from filename
	b) Resize to 256x256
	c) Normalize to range [0,1]
	d) Repeat (a)-(c) for label image, but also adding channel dimension of length 1
4. Load a single image/label and print meta-data to ensure steps 1-3 are working (Output below).
Image shape:  (256, 256, 3)
Label shape:  (256, 256, 1)

Data info:
height: 256
width: 256
channels: 3
![Sample image/label](/images/image_label_example.png)
5. Call model-building function from model.py.
6. Create binary segmentation DICE loss function/metric (plus numpy version for use in predictions).
7. Compile model with Adam optimizer, DICE loss function, and DICE metric.
8. Train model for 100 epochs, with batch size 16 (using .batch), and Tensorboard callbacks enabled.

### Improved Unet Model
**File: model.py**
Context pathway:
*

Localization pathway:
*

Architecture diagram:
![Improved U-net architecture](/images/architecture.png)

### Evaluation
**File: driver.py**
1. Generate training/validation accuracy plot (with DICE)
2. Use model.evaluate on test set
3. Use model.predict on test set (The average of this is used as the ultimate performance metric)
4. Plot 3 predictions along with ground-truth
The output of this is given in the Output section below.

## Dependencies
- Tensorflow (GPU)
- Glob
- Numpy
- Matplotlib
*- There should be a directory named 'tb' in the working directory*

## Training/validation/test split
20% of the dataset is set aside as the test set.
Of the remaining 80%, 20% (relative) is set aside for the validation set (i.e. 16%).
The remaining images make up the training set (i.e. 64%).

As the dataset is not particularly large, a decent portion of the images need to be designated to validation/test sets. 20% should be fine for this.

## Output
Average DICE score (test set predictions): x%

Sample predicted segmentation:
![Predicted segmentations](/images/predictions.png)

Training accuracy plot:
![Training accuracy plot](/images/training_plot.png)

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
[2] O.  Ronneberger,  P.  Fischer,  and  T.  Brox,  “U-net:  Convolutional  networks  forbiomedical  image  segmentation,”  in *International  Conference  on  Medical  Image Computing and Computer-Assisted Intervention*. Springer. 2015. pp. 234–241
[3] K. He, X. Zhang, S. Ren, and J. Sun, “Identity  mappings  in  deep  residual  networks,” in *European  Conference  on  Computer  Vision*. Springer. 2016. pp. 630–645.
[4] B. Kayalibay, G. Jensen, and P. van der Smagt, “CNN-based  segmentation  of medical imaging data,” Jan. 2017. [Online]. Available: https://arxiv.org/1701.03056