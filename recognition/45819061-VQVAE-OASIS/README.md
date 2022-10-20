# Vector Quantized - Variational Autoencoder for Generation of OASIS Brain data


Here we construct a Vector Quantized - Variational Autoencoder (VQ-VAE)model trained on the OASIS brain dataset to construct a generative model which can reproduce and generate brain scan images by sampling a discrete latent space much smaller than the desired images.

# Problem
Development in computer technology in recognising and classifying brain disease is a growing field which aims to develop effective computer models that can recognise and classify information in brain scans to identify problems and characteristics of a patients brain. A limitation in the effectiveness of this technology currently stems from an insufficient amount of data to train these classification models and thus the models that are produced are undertrained and ineffective. We use a VQ-VAE as a way of learning the structure and characteristics of brain scans and encoding into a smaller compact latent space. We learn patterns and structures of this latent space and train a generative model that generates clear and new brain scans which can be used to train these classification models. 

# The Model
The model we train is a VQ-VAE consisting of an encoder feeding into a vector quantizer layer whose output then feeds into the decoder. The encoder and decoder are both made of to convolutional blocks and two residual layers. The convolutional layers are 4x4 windows with stride 2 and reduce the image data by a factor of four before passing to the residual layers. We use filter sizes 32, 64. Next, the residual layers are two convolutions (3x3 and 1x1) with filter size 32 and leaky relu activations between. The output of the residual block is the sum of the out put of this convolution wth the original data. Vector Quantizer layer consists of a codebook of embedding codes, the VQ layer takes the output of the encoder and computes relative distance to these embeddings to find the images supposed place in the latent space. VQ can be thought of as being given the identified key characteristics of the image by the encoder and then the VQ assigns the output the indices where such information is stored in the latent. Finally a decdoer takes a set of odewords from the latent space and via 2 transposed convolutional layers and residual blocks the image is rebuilt. During training the VQVAE attempts to maintain the integrity of its vector quantisation of the latent space and its reproduction of the image. 
For generation of images we train a PixelCNN on the latent space discovered by the VQVAE to sample the latent space and discover new codes t pass to the decoder to generate realistic brain scans.

# Requirements
Although versioning may not be strict this is what was used in this case.
- tensorflow = 2.10.0
- tensorflow-probability  =  0.18.0 
- tqdm       =               4.64.1
- matplotlib  =              3.6.1
  
# Training
We train the models with Adam optimizers tracking commitment loss, codebook loss and reconstruction loss in the case of the VQVAE, and categorical entropy in the case of the pixelcnn. Filter sizes for each convolutional layer in teh sytem must be sufficiently large to avoid the model training to an unusable state as was the case below. 

![0.5](losses.png) ![](fig9.png)
