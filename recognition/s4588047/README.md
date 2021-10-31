

#  Vector-Quantized Variational Autoencoder (VQ-VAE) for _OAI AKOA knee dataset_

_Author_ : _Yousif Al-Patti_

_Student No_ : _45880472_

_Dataset_ : _OAI AKOA knee dataset_

_Email_ : s4588047@student.uq.edu.au

##  Description VQ-VAE
Vector Quantized Variational Autoencoder (VQ-VAE).
In essence, the VQ-VAE learns a discrete latent representation and using the encoder the input data converted into discrete codes by increasing the number of filters on each layer. 
Conv2D(hidden//2) -> Conv2D(hidden) -> Conv2D(hidden) -> ResidualStack(hidden)
This creates a bottleneck stage and unlike a normal autoencoder, the bottleneck is quantised by adding a discrete codebook thought the VectorQuantized layer such that the encoder output is associated with the closest vector in the codebook network (C. Snell, 2021). Now the output of this layer is fed to the decoder which in turn consist of reversed layers to the encoder
Conv2D(hidden) -> ResidualStack(hidden) -> Conv2DTranpose(hidden//2) -> Conv2DTranpose(1)
Two of the concepts used in addition to what was proposed in the original paper are the ResidualStack and a Pre-VQ layer convolution. The ResidualStack acts as a skip connection to provide a better flow of information from the initial input the output of the convolutional layers. Furthermore, the Pre-VQ layer is added with number of filters being the capacity of the information bottleneck to ensure the data fit the required shape before being passed to the VQ layer.
Now this is the main model for the network, but it also contains a Pixel CNN model which trains on the codebook indices stored in the VQ layer with encoded input and it generates a new codebook representation by sampling from the categorical distribution of the Pixel CNN output. Now the output of the new model can be passed to the decoder to generate a new image that is not a reconstruction of a previous image directly.
![VQ-VAE Model structure used (Aaron van den Oord, et al., 2018)](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/resources/model_overview.png)

##  Dependencies
The dependencies are also specified in requirements.txt.
tensorflow-gpu 2.6.0
tensorflow-probability 0.14.1
matplotlib 3.4.3

### Tensorflow: 
To install Tensorflow, use the following command:
```
pip install tensorflow-gpu
```

### Tensorflow Probability
To install matplotlib, use the following command: 
```
pip install tensorflow-probability
```

### Matplotlib
To install matplotlib, use the following command: 
```
pip install matplotlib
```

##  Test Script
### Data preperation
Using data_loader.py file, the images are loaded into a TF BatchDataset object using `tf.keras.preprocessing.image_dataset_from_directory`
The data are loaded as a batch of 256 and split 80% for training and 20% for validation. 
The ratio of data splitting has been chosen because it is commonly used as it follows Pareto principle "roughly 80% of consequences come from 20% of causes". Also, this process helps avoiding overfitting the model and provide a larger enough set for validation as recommended by the paper implementation (T. Hennigan, 2018).

Later the data are normalised in process_data by converting them into greyscale and dividing by 255. to scale the data between 0 and 1 floating point values.

### How to run
Once the requirements are met, executing the test script out of the box would load the data and plot a sample image from the training dataset. Furthermore, 10 reconstruction plots will be printed after the model has been trained and follows a plot of the losses. Moreover, the Pixel CNN network is trained and an image sample of one of the validation images is printed along with its code indices. Lastly, a new code indices are generated and passed to the decoder resulting in an image of the new code and decoder output of that code.

##  Result
Reconstructions after 20 epochs
![20 epochs original and reconstruction](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/20%20epochs/Figure%202021-10-27%20120358%20%280%29.png)
Reconstruction after 100 epochs
![100 epochs original and reconstruction sample](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/100%20epochs/Figure%202021-10-31%20004610.png)
![100 epochs loss plot](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/100%20epochs/Figure%202021-10-31%20004701.png)
![100 epochs codebook indices sample](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/100%20epochs/Figure%202021-10-31%20004725.png)
![100 epochs Pixel CNN output reconstruction sample](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/100%20epochs/Figure%202021-10-31%20004827.png)
![100 epochs Pixel CNN output reconstruction sample](https://github.com/yousifpatti/PatternFlow/blob/topic-recognition/recognition/s4588047/outputs/100%20epochs/Figure%202021-10-31%20114344.png)
Note: the quality of the images output for the VQ-VAE do improve with more epochs and they get finer details and clearer. The expected images would ideally look like the original image or slightly blurry and this can be seen happening within 100 epochs. Furthermore, for the Pixel CNN reconstruction it is possible to see slight shadows around the image produced showing that the model layers are outputting the file details for different images. With further training for the Pixel CNN and adjustment of the training size it would be possible to produce better codebook indices that can be decoded to a more familiar images such within the _OAI AKOA knee dataset_ field.
##  References
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    Tom Hennigan, VectorQuantizer layer, 2018
    
    https://keras.io/examples/generative/vq_vae/
    Sayak Paul, Vector-Quantized Variational Autoencoders, 2021
    
    https://arxiv.org/abs/1711.00937
    Aaron van den Oord, et al., Neural Discrete Representation Learning, 2018
    
    https://ml.berkeley.edu/blog/posts/vq-vae/
    Charlie Snell, Understanding VQ-VAE, 2021
