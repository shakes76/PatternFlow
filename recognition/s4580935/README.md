# Vector Quantized Variational Autoencoder (VQVAE)
## Generative model of the OASIS brain dataset
</br>

Task: Create a generative model of the OASIS brain data set using VQVAE that has a reasonable clear image and a structured similarity (SSIM) of over 0.6. The model firstly  takes a basic vqvae with some convolutional layers to reconstruct brain images from a dataset and secondly uses a PixelCNN prior to attempt to generate "new" brain images. 
Examples of the origional, reconstructed and "new" brain images generated from the models can be found below.

## VQVAE and how it works
</br>

Auto encoders come in many forms, ranging from their most basic and initial implementation , to a variational auto encoder, then a vector quantized variational autoencoder and even further. Proposed by Van der Oord er al in the Neural Discrete Representation Learning paper last revised in 2018 [1]. Like a typical variational auto encoder the VQVAE is composed of three components; An **Encoder**, **Decoder** and a **Latent Space**. 

**Encoder**: Models a categorical distribution, from which you get integral values. Done through the use of a convolutional network that extracts the features of the supplied input images. 

**Decoder**: Takes in the output of the latent space (Vector Quantization layer), and attempts to recreate the initial input given to the encoder. Uses Convolutional network to upsample (reverse the effects of the encoder) and reconstruct the origional input image from a simplified/compressed input.

**Latent Space (Vector Quantization Layer)**: This layer takes the output from the encoder as its input and embeds selected sata based on distance and supplied a corresponding compressed output which is taken as the input for the decoder. This layer maintains a codebook of embeddeding vectors of a set dimension equal to the filters in the encoders output. 
[2]
<p align="center">
<img src="./img\1_miNfFc9qT5PrS7ectJa_kw.png">
</p>
[3]

# PixelCNN
</br>

# Model Results
</br>
The model generated was trained on the OASIS brain dataset. This dataset can be found with some preprocessing done on the [COMP3710 Blackboard Page](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA). To use this data you will need to download and extract the zip file and ensure that the paths specified in train.py file for the variables train_images, test_images and validate_images point to the corresponding extracted non-segmented files. 

The images found in these files are 256x256 greyscale pixel png's. In train.py using functions from dataset.py the images are extracted and processed such that they are normalised to values between 0 and 1 and the train and validate sets are combined into a single set of 10784 brain images which are used to train the model. the remaining 544 test images are used for evaluating the accuracy of the model.

The modules.py file stores the class and functions for the VQVAE model and is called when running train.py. The encoder and decoder are both simple in their construction involving a convolutional layer with filter numbers of 32 and 64 and transposed convolutional layer with 64 and 32 respectively. The summary of the VQVAE model can be seen as follows:
<p align="center">
<img src="./img\vqvae.png">
</p>

After Training the model on the combined train and validate set (named Oasis) for 30 epochs using batch sizes of 128 a model was generated. The reconstructed loss over the epochs can be seen in the following graph.
<p align="center">
<img src="./img\Reconstruction Loss.png">
</p>
As it can be seen the model starts performing quite badly which is to be expected but after around 20/30 epochs the loss begins to plateau.

To test the Model the test set was used to determine how well the model can reconstruct images. The goal of the model was to be able to reconstruct brain images with a structural simalarity of 0.6 or more and based on the 544 test images supplied the model returns a structured simalarity of 0.75, with the lowest similarity at 0.68 and highest at 0.80. Examples  of both the origional test images and their reconstructed images and their codebook images can be seen as follows:
<p align="center">
<img src="./img\1.png">
</p>
<p align="center">
<img src="./img\2.png">
</p>
<p align="center">
<img src="./img\3.png">
</p>
<p align="center">
<img src="./img\c1.png">
</p>
<p align="center">
<img src="./img\c2.png">
</p>
<p align="center">
<img src="./img\c3.png">
</p>


The next part of the model that was generated is the PixelCNN model that is suppose to try and generate new brain images and then reconstruct them. This model was significantly more resource intensive to train and required colab pro premium gpu's and high ram and still took time. The code for this model was taken from the corresponding keras tutorial on VQVAE's and due with the current parameters performs quite poorly when it comes to creating "new" brains [4]. The summary for the model is as follows:
<p align="center">
<img src="./img\pixel.png">
</p>
The model is starting to head in a direction where the images are starting to have either the general shape of a brain or general patterns of a brain, however overall perform quite poorly when it comes to actually generating a "new" and recognisable brain image. Potential changes that could be further made to the model to try and make the images more realistic would involve increasing the number of residual blocks and pixcelcnn layers. By doing this the number of parameters the model can train on would increase which could result in a more realistic looking brain. Other hyper-parameters that could be manipulated is the number of filters in each layer and kernel size. As it currently is the "new" brains look as follows:
<p align="center">
<img src=".\img\g1.png">
</p>
<p align="center">
<img src=".\img\g2.png">
</p>
<p align="center">
<img src=".\img\g3.png">
</p>

From these images it is quite clear to a human that these images are not "brains" and as a result if the model were to be used to generate images of new brains the parameters of the PixelCNN would need to be modified. However the model does perform well at reconstructing code book images even the "new" ones it generates. 

# Dependencies
</br>
 - Tensorflow version 2.5 or higher (used 2.9.2)
 - Tensorflow Probability version 0.16.0 (possible to install with the following command):
!pip install -q tensorflow-probability
 - matplotlib
 - numpy 1.21.6

Please Note that testing and building of this model was done in google colab using colab pro, the model was run using access to the high performance ram and gpu's and cant all be run using standard colab without running out of memory. It is possible that the models could be run in standard colab if significant changes were made to the number of images used for training and how long models can be run for. Furthermore using colab all the dependencies such as tensorflow, matplotlib, numpy etc should be supplied for the user. 
Note if the files are run using colab changes will need to be made to the file paths for accessing the images and you will likely need to allow access to google drive. For an example of how to run the files as a standalone script see the Report.ipynb file.


## References
[1]Oord, A.van den, Vinyals, O. and Kavukcuoglu, K. (2018) Neural Discrete Representation Learning, arXiv.org. Available at: https://arxiv.org/abs/1711.00937. 
[2]Yadav, S. (2020) Understanding vector quantized variational autoencoders (VQ-VAE), Medium. Medium. Available at: https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a. 
[3]Park, S. (2021) An overview on VQ-VAE: Learning Discrete Representation Space, Medium. Analytics Vidhya. Available at: https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337. 
[4]Team, K. (2021) Keras documentation: Vector-quantized variational autoencoders, Keras. Available at: https://keras.io/examples/generative/vq_vae/ (Accessed: October 17, 2022). 
