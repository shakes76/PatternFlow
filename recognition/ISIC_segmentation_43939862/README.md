# ISIC Image Segmentation Using Improved Unet

The model used in this project, based on the ConvNet from [1], is an improved version of Unet.[2] 
This improved Unet uses the main aspects of the original Unet – an autoencoder with long skip connections – 
as well as a few additional features. Firstly, they use residual blocks in the context pathway (encoder), 
inspired by [3]. By adding short skip connections (identity mappings in this case, shown to be optimal by [4]) 
from the input to the output of a block, connected with an element-wise sum, the gradient can flow more 
freely and the gradient-vanishing problem is mitigated, allowing for deeper networks.[5] Furthermore, the 
authors used deep supervision, encouraging the network to produce good segmentation maps throughout the 
decoder by adding those segmentation maps directly to the output.[6] Among other novel changes were 
normalization procedures, differing numbers of feature maps, and data augmentation. Many of these details 
were not relevant to this project, and thus were not included. In this project, the improved Unet is applied 
to the ISIC 2018 (task 2) dataset, with the goal of segmenting the skin cancer images with an average DICE 
score of at least 80% on the test set. This dataset has been pre-processed and includes 2594 RGB images of 
varying resolutions, along with their ground truth segmentations.
<br/>
<br/>
<br/>
## The Algorithm
### Pre-processing and model-handling
**File: driver.py**
1. Read image/label filenames and split into training/validation/test sets.  
	- Images/labels are in directory 'ISIC2018_Data' outside the repo, in folders called 'ISIC2018_Images' and 'ISIC2018_Labels' respectively.
2. Format sets into tensorflow datasets and shuffle each of them.
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

![Sample image/label](/recognition/ISIC_segmentation_43939862/images/image_label_example.png)


5. Call model-building function from model.py, which returns the Keras model directly.
6. Create binary segmentation DICE loss function/metric (plus numpy version for use in predictions).
7. Compile model with Adam optimizer, DICE loss function, and DICE metric.
8. Define the learning rate schedule.
9. Train model with batch size 16 (using .batch).

### Improved Unet Model
**File: model.py**  
As with the original Unet, the network is an autoencoder made up of a context aggregation pathway, 
which extracts relevant features from the input at varying levels of abstraction, and a localization pathway, 
which recombines those features, as well as features directly from the encoder (via long skip connections) 
to generate the desired output. The architecture of these two pathways will be discussed in detail.

*Context pathway:*
The context pathway is made up of a series of down-samplings via 3x3 filter 2x2 stride convolutions, 
followed by a context module. The context modules are pre-activation residual blocks, containing 2 sets of: 
Batch normalization, leaky relu activation (hence pre-activation), and a 3x3 convolution with a kernel regularizer. 
There is a r=0.3 dropout layer between these sets. The short residual skip connections are added to the output 
of these sets via element-wise sum. Note that the very first layer (after the input layer) is a normal 3x3 
convolution rather than a down-sample.

*Localization pathway:*
The localization pathway is made up of a series of up-samplings via 2x2 upsamples and 3x3 convolutions halving the feature maps (with batch
normalization and leaky relu activation), concatenation with the corresponding layer from the encoder via long skip connections, 
followed by a localization module. The localization modules contain a 3x3 filter convolution with a kernel regularizer, 
batch normalization, and leaky relu activation, followed by a 1x1 convolution (to halve the number of feature maps), 
batch normalization and leaky relu activation. Furthermore, the outputs of the 3x3 convolutions are also up-sampled, 
added with each other (after reducing feature maps to 1 via 1x1 a convolution), and added to the output, 
allowing for deep supervision.

Architecture diagram:  
![Improved U-net architecture](/recognition/ISIC_segmentation_43939862/images/architecture.png)

### Evaluation
**File: driver.py**
1. Generate training/validation accuracy plot (with DICE https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
2. Use model.evaluate on test set
3. Use model.predict on test set (The average of this is used as the final performance metric)
4. Plot 3 predictions along with ground-truth  
The output of this is given in the Output section below.
<br/>

## Dependencies
- Tensorflow (GPU)
- Glob
- Numpy
- Matplotlib  
- There should be a directory named 'tb' in the working directory
<br/>

## Training/validation/test split
20% of the dataset is set aside as the test set.
Of the remaining 80%, 20% (relative) is set aside for the validation set (i.e. 16%).
The remaining images make up the training set (i.e. 64%).

As the dataset is not particularly large, a decent portion of the images need to be designated to validation/test sets. 20% should be fine for this.
<br/>
<br/>
<br/>

## Output
*Note: Running the default learning rate for 200 epochs finds a low validation loss quickly, with damped oscillations after that (see extra image in /images). So for final training, 
I used a learning rate schedule with normal initial rate and aggresive decay, together with a StopEarly callback. This found a solution equally as accurate on the validation set in only 16 epochs*

**Average DICE score (test set predictions): 0.8419404442018784**

Training accuracy plot:  
![Training accuracy plot](/recognition/ISIC_segmentation_43939862/images/training_plot.png)

Sample predicted segmentation:  
![Predicted segmentations](/recognition/ISIC_segmentation_43939862/images/predictions.png)
<br/>
<br/>

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1  
[2] O.  Ronneberger,  P.  Fischer,  and  T.  Brox,  “U-net:  Convolutional  networks  forbiomedical  image  segmentation,”  in *International  Conference  on  Medical  Image Computing and Computer-Assisted Intervention*. Springer. 2015. pp. 234–241  
[3] F. Milletari,  N. Navab,  and S.A. Ahmadi. "V-net:  Fully convolutional neural networks forvolumetric medical image segmentation," 2016. [Online].  https://arxiv.org/1606.04797  
[4] K. He, X. Zhang, S. Ren, and J. Sun, “Identity  mappings  in  deep  residual  networks,” in *European  Conference  on  Computer  Vision*. Springer. 2016. pp. 630–645  
[5] B. Kayalibay, G. Jensen, and P. van der Smagt, “CNN-based  segmentation  of medical imaging data,” Jan. 2017. [Online]. Available: https://arxiv.org/1701.03056  
[6] H. Chen, Q. Dou, L. Yu, and P.A. Heng. "Voxresnet:  Deep voxelwise residual networks forvolumetric brain segmentation," 2016. [Online]. Available: https://arxiv.org/1608.05895v1
