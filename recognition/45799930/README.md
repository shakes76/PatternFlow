# Segmentation of the ISIC Dataset with the Improved UNet

---

Author: Jessica Sullivan

Student id: 45799930

Assignment: COMP3710 Report Semester 2, 2022 

---


## Description of the Algorithm and Problem it Solves

The algorithm is based upon the improved UNet structure. This

---



## How the Algorithm Works

---



## Visualisation

---



## Dependencies

### Versions required:

```commandline
Tensorflow: 2.10.0
Matplotlib: 3.5.3
```

### Address reproducibility:

To ensure that the code can run, you will need to download the training dataset and the truth training dataset where you should download the one in the first row which have the binary masks in PNG format. These can be downloaded from [here](https://challenge.isic-archive.com/data/#2017). Once downloaded these folders should be moved to the recognition/45799930 directory, keeping the same names that where made when created. Therefore the directories that should have been added are:

* ISIC-2017_Training_Data
* ISIC-2017_Training_Part1_GroundTruth

---



## Justification

### Specific Pre-Processing

There was some pre-processing done on the data, to ensure that the images are downloaded and processed correctly, that they where the same size and that the colouring is correct. This was all done in the `dataset.py` file. Once the image was read by using the correct pathway, if it was decoded depedning on the file type (jpeg for images and png for the truth images). This was then processed to make sure that all images where of the size (256, 256). After ensureing that we where only dealing with elements that where of type `tensorflow.float32` the image was normalised by dividing by 255. 

### Training, Validation and Testing Splits

---



## Examples

### Example Input:

### Example Output:

---
