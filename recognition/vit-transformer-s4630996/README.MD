# Section Navigation
1. [Description of Algorithm](#description-of-algorithm)
2. [Problem It Solves](#problem-it-solves)
3. [How It Works](#how-it-works)
4. [Figure/Visualization](#figure--visualizaton)
5. [Dependencies](#dependencies)
6. [Reproducibility of Results](#reproducibility-of-results)
7. [Example Inputs/Outputs](#example-inputs--outputs)
8. [Plot of Algorithm](#plot-of-algorithm)
9. [Pre-processing](#pre-processing)
   - Preventing Data Leakage
   - Cropping to Improve Computational Efficiency
   - Expanding the Dataset with Augmentation
10. [Training and Validation Splits](#training--validation-splits)
11. [References](#references)



# Vision Transformer for ADNI Images

#description-of-algorithm
## Description of Algorithm


This is an implemention of a Vision Transformer (ViT) which follows as closely as possible the model presented by Dosovitskiy et al $^1$ in their 2021 paper *An Image is Worth 16x16 Words: Transformers For Image Recognition at Scale*.     

![This is an image](./images/vit.png)
(Image Reference)

This implemenaton of a ViT has been tailored for use with data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).  This data was provided on the COMP3710 Blackboard site in a zip format.  It is unknown as to what extent this data matches that available directly from ADNI.

The ADNI images are 240x256 greyscale jpeg files.  Details on image pre-processing and image directory structure can be found in the [Pre-processing](#pre-processing) section.  

*ViT Stage 1: Create Embedding Vectors from Input Images*

The ViT takes as input a sequence 1D arrays, each of which is a flattened "patch" extracted from an image.  For example, a 240 x 240 image with a patch size of 16 would create 15 x 15 = 225 patches each having a length of 256 once flattened.  Each of the flattened patches is then mapped onto *D* dimensions (the projection dimensions).  These projections are referred to as the patch embeddings.

A class token of the same dimensions as the flattened patches is then prepended to the patch embeddings.  Continuing with the example from above there would now be 226 1D inputs (225 flattened patches plus the class token).  

The final step in creating the embedding vectors is to add a position embedding to the patch embeddings.  The embedding vectors are now ready for input into the Transformer Encoder.

*Stage 2: Learn Spatial Relations Between Patches with Transformer Encoder*

The embedding vectors are now passed through one or more layers of the Transformer Encoder.  Each layer of the transformer encoder is constructed from two blocks.  The first block consists of layer normalization followed by multi-head self attention, and the second block consists of layer normalization followed by a multilayer perceptron.  For each block a residual connection is applied.

*Stage 2: Classify with a Multilayer Perceptron using the Class Token*

The class token is now extracted from the Transformer Encoder output and fed into a multilayer perceptron("The MLP Head") which consists of two or more hidden layers.  The output of the MLP Head is used to predict the sample classification.

[Back to Top](#section-navigation)
# Problem it Solves

 Since their introduction in 2017 $^3$ self-attention based transformers have  been the model of choice for state of the art natural language processing.  At the same time convolutional neural networks have dominated computer vision and image classification.  The use of vision transformers in image classification is an opportunity to extend the success of attention based transformers into the domain of computer vision.  Such an extension is enticing due to the computational efficiencies and scalabilities that transformers provide. Dosovitskiy et al's paper was groundbreaking in that it demonstrated that self-attention based transformer architectures could compete with state of the art convolution based architectures in many image recognition tasks. $^1$

One reason why CNNs are so successful image recognition is their ability to identify local features in the image that are translation invariant.  The inductive bias of the CNN architecture promotes both identification of these features and their assembly into a hierarchical structure.  In contrast, attention transformers have been successful due to their ability to provide global context.  For example, the word "bank" has several different meanings depending on the context in which it is used and transformers are proficient at identifying this context.  It is therefore significant that the contextual approach (eg attention based transformers) has been found to be successful in image recognition where hierarchical feature maps have been dominate.

This Vision Transformer implementaion attempts to solve the problem of Alzheimer's Disease diagnoses using a 2D image of a patient's brain.  Alzheimer's Disease is a neurological disorder that affects tens of millions of people worldwide and its early detection is critical to mitigating the adverse impacts of the disease on patients and their families. $^4$ 

[Back to Top](#section-navigation)

---

# How it Works
The key steps for using this model are below.  

1. **Download and unzip data** 
2. **Create environment**
3. **Prepare the data** 
   - Create new folder structure
   - Create validation and training from train to preclude leakage
   - Augment?
   - Crop the images
4. **Update config.py**
5. **run python train.py**
6. **Load model weights and make predictions by running python predict.py** 

Each of the steps above is now reviewed in more detail.

#### 1. <u>Download and unzip data</u>

The ADNI images are received in a zip format.  Once extracted, the images are placed in the following file structure:

```bash
├───test
│   ├───AD
│   └───NC
├───train
│   ├───AD
│   └───NC
```
The training set contains 19360 images, of which 9360 are AD (classified Alzheimers) and 10000 are NC v.  The test set contains 9000 images of which 4460 are AD  and 4540 are NC.  


#### 2. <u>Create environment</u>

It is necessary to create two different environnments, one for image pre-processing and another for the Vision Transformer.  The two environments are recommended due to conflicts in dependencies that could exist between the Augmentor package and other Python image modules.  Augmentor uses Pillow, which is a fork of PIL, but both packages cannot exist simultaneously.

The ViT is developed in a Tensorflow environment.  Tensorflow 2.9 is recommended for compatibility with Tensorflow Add Ons and Keras CV.  For installation of the Tensorflow environment it is strongly recommended to use a Windows 11 based PC and follow the guidance available here:

https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb

A listing of required modules is provided below in [*Dependencies*](#dependencies).


[Back to Top](#section-navigation)

---

# Figure / Visualizaton

- training loss and accuracy (best, anomolies)
  - validation accuracy above training accuracy
  - good loss / accuracy plot, terrible test results
  - stagnation
  - sgd vs adam for same parameters
- table of experiments

- 

<u></u>
<u></u>
<u></u>

 
[Back to Top](#section-navigation)

---

# Dependencies

### Image Preprocessing
- augmentor
- python 10

### ViT
- python 3.9
- matplotlib
- tensorflow 2.9
- numpy
- keras_cv
- tensorflow_addons

[Back to Top](#section-navigation)

---

# Reproducibility of Results
- initial selection of validation set
- seed in image_dataset_from_directory
- hyperparams --> obviously stochastic by nature

[Back to Top](#section-navigation)

---

# Example inputs / outputs
- input: brain image?
- output: test results?

[Back to Top](#section-navigation)

---

# Plot of Algorithm
- training loss and accuracy?

[Back to Top](#section-navigation)

---


# Pre-processing
1. create folders
2. prevent leakage
3. crop
4. augment


The ADNI images are received in a zip format.  Extracted images are placed in the following file structure:

```bash
├───test
│   ├───AD
│   └───NC
├───train
│   ├───AD
│   └───NC
```

The test set contains 9000 images of which 4460 are AD and 4540 are NC.  The training set contains 19360 images, of which 9360 are AD and 10000 are NC.  

Here is image "218391_78.jpeg" from the AD folder:

![This is an image](./images/218391_78.jpeg)

Here is image "839474_78.jpeg" from the NC folder:

![This is an image](./images/839474_78.jpeg)

The files names appear to be formatted such that the numbers separated by "_" has some significance.  An assumption can be made that the first number is a patient ID and the second number is an image ID for this patient.  This assumption is supported by the fact that each ID is associated with 20 images.  Thus the 9360 training images for class "AD" contain images from 468 unique patients.

#### Potential for Data Leakage

This insight into the provenance of the filename has important implications for the selection of a method to generate a validation set.  If patient IDs are not taken into consideration and validation images are selected at random from the training set, then it would be possible for the images of one or more patients to appear in both the training and validation ("data leakage").  Thus a python script was created to generate the validation set taking into account the patient ID so that no images for any single patient are found in both the training and validation set.  

#### Additional Observations Regarding Images

Three additional observations regarding the images have implications for our pre-processing.  The first is that the images have the shape 256 x 240 (w x h).  The vision transformer requires square images from which to make patches so each image will need to be cropped.  Furthermore we can see that there is lots of empty space around each brain image.  Assuming the empty space will not play a role in classification performance, we can crop the images to a smaller size.  The benefit to the model is computational - for the same patch size, fewer patches will need to be processed for each image.  However, the brains will have variation in size and relative position on the image so care must be taken to select the minimum size that will reliably capture all brain images.  After some tinkering, an image size of 200 x 200 was found to significantly reduce empty space while also ensuring no brains were cropped.

#### Mitigating The Impact of Dataset Size on ViT Performance




This is a grayscale image with the dimensions 256 x 240.   
- lots of empty space
- orientation
- brightness
- 




[Back to Top](#section-navigation)

---

# Training & Validation Splits


[Back to Top](#section-navigation)

---



# References

$^1$ An Image is Worth 16x16 Words: Transformers For Image Recognition at Scale

$^2$ https://adni.loni.usc.edu/

$^3$ Vaswania Attention is all you need

$^4$ https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447

Class token impl:
https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html

Class token impl:
https://medium.com/geekculture/vision-transformer-tensorflow-82ef13a9279


ViT architecture
https://keras.io/examples/vision/image_classification_with_vision_transformer/

$n_{1}$

[Back to Top](#section-navigation)
 
 ---

To do:

footnotes? 

[^2].

[^2]: 

model summary from tensorflow

