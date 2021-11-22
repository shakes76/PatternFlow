# README (Khac Duy Nguyen, 46225744)
Task 5. Classify laterality (left or right sided knee) of the OAI AKOA knee data using the Perceiver transformer set having a minimum accuracy of
0.9 on the test set. [Hard Difficulty]

# Perceiver Transformer
![Architecture](https://github.com/camapcanhcut/PatternFlow/blob/topic-recognition/recognition/s4622574/resources/architecture.png?raw=true)

# Environment Setup
* `conda env create -n env`
* `conda activate env`
* `conda install pip`
* `pip install tensorflow`
* `pip install tensorflow-addons`
* `conda install matplotlib`

# Dependencies
* `TensorFlow`
* `Matplotlib`
* `Keras`
* `Numpy`

# Implementation
* Download OAI AKOA knee dataset and put into the same folder as source code
The implementation is divided into 4 parts:
* `main.py` :  Dataset preparation, initialize Perceiver Transformer, then train model from scratch
* `perceiver.py` :  Architecture of Perceiver Transformer
* `fourier.py` :   Encoding of image data
* `attention.py` :  Create attention mechanism

# Dataset Preparation
There are 101
unique patient in the dataset, while the total number of images is 18680. We will preprocess data so that we can ensure images of the same person will not in both trainning set and testing set. This will solve the overfitting problem (generalization).

# Experiment
Main executable can be run from main.py which calls: 
* `python main.py`

Testing Accuracy:
![Accuracy](https://github.com/camapcanhcut/PatternFlow/blob/topic-recognition/recognition/s4622574/resources/result.png?raw=true)