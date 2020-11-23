# Improved UNet for ISICs Data
__**Created by Matthew Costello, 45314838**__

This is my implementation of ['the improved UNet'](https://arxiv.org/pdf/1802.10508v1.pdf), adapted for 2D binary class segmentation. It is a convolutional neural network that outputs an original image as a mask, segmented into white and black. Specifically, the parameters of this model were tuned for the [ISIC 2018 challenge data for skin cancer](https://challenge2018.isic-archive.com/) - this model is capable of segmenting skin lesions with an **average [Dice similarity coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of about 0.95** (example outputs visualised below).

The model works by first increasing the spatial resolution and decreasing the representation of dimensional features going down the 'U'- this allows it to learn the feature mapping by converting the image into a vector. Then going back up the 'U', the vector is converted back into an image using the previously learnt feature mapping, resulting in a segmented image with the same resolution as the original input image. Dice distance (1 - DSC) is used as the loss function, which was particularly well-suited to this data set due to the differences in relative lesion sizes that would need to be segmented. For example, if the skin lesion was relatively small, a large amount of black space would be correct, although this could result in an inflated accuracy while the small white area itself is not accurately segmented. DSC allows for proper foreground accuracy regardless of relative sizes (imbalanced classes).

With the use of my generators to allow for fast on-the-run data inputs, I was limited to only being able to create a training data generator and a testing data generator by the tf.keras implementation. However, the lack of a validation set is more justifiable in this case as I was not trying to determine which model I would use for this problem; I already had a given model (the improved UNet) that I based my implementation on. Furthermore, my training set was of a reasonable size (2000+), and I was focusing on implementing the referenced model accurately, rather than using the test set for validation set purposes. My training/test sets were shuffled and split at a standard amount of 80/20, respectively.

## Dependencies
My environment.yml file is provided in this directory for ease of use with Anaconda. The following are required:
1. Python 3.7
2. Tensorflow-gpu 2.1
3. Numpy
4. Matplotlib
5. Tensorflow_addons 0.9.1<sup>1</sup> (NOTE: This isn't included within Conda, but pip can be used to add it to the Conda env - v0.9.1 required for TF 2.1)

**NOTE: Data must be placed as a FOLDER within either data/mask or data/image - not directly into these directories, for the generators to read.**

## Visualisation
![Figures](resources/visuals.jpg?raw=true "Title")


\[1\] 'tensorflow-addons' is an officially supported repository implementing new functionality. More info at https://www.tensorflow.org/addons. Version 0.9.1 is required for TF 2.1. TFA allows for a InstanceNormalization layer (rather than a BatchNormalization layer), as was implemented in the referenced 'improved UNet'. This layer is necessary due to the usage of my small batch-size of 2, which can lead to "stochasticity induced ...\[which\]... may destabilize batch normalizaton" - F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge” Feb. 2018. \[Online\]. Available: https://arxiv.org/abs/1802.10508v1. While BatchNormalization normalises across the batch, InstanceNormalization normalises each batch separately.
