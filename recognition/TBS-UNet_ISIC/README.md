# Segmentation of ISIC data set with U-Net
COMP3710 Final Report Task

Tristan Samson

## Algorithm Description
The model built in model.py and run in driver.py are an implementation of the original
[U-Net](https://arxiv.org/abs/1505.04597) semantic segmentation network. This is a convolutional neural network often applied
to biomedical image segmentation. The implementation of U-Net provided in model.py is
application-neutral. It has been applied in driver.py to provide segmentations of the
ISIC 2018 challenge data for skin cancer. Segmentations distinguish between regions
of the image belonging to potentially cancerous legions and regions belonging to regular skin.

The U-Net has been implemented exactly as it was described by Ronneberger, et al. A
figure of the network structure is provided below.

<img src="figures/unet_structure.png" alt="U-Net Network Structure" width="600"/>

Talk about: 

General network structure - encoder-decoder

Choice of loss function - directly optimising dice coefficient


## Learning Behaviour



## Example Outputs
### Training behaviour
<p float="left">
<img src="figures/train_accuracy.png" alt="Training Convergence (Accuracy)" width="400"/>
<img src="figures/train_loss.png" alt="Training Convergence (Loss)" width="400"/>
</p>
The average Dice Coefficient over the test set is: 0.82

### Example Outputs
Example segmentation 1 (Dice Coefficient = 0.83)
<p float="left">
<img src="figures/input_1.png" alt="Example 1 (Input)" width="400"/>
<img src="figures/seg_1.png" alt="Example 1 (Segmentation)" width="400"/>
</p>
Example segmentation 2 (Dice Coefficient = 0.67)
<p float="left">
<img src="figures/input_2.png" alt="Example 2 (Input)" width="400"/>
<img src="figures/seg_2.png" alt="Example 2 (Segmentation)" width="400"/>
</p>