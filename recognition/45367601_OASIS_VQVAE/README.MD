# Generative model of the preprocessed OASIS brain dataset 
The vector quantified variational auto encoder (VQ-VAE) is a variation on the traditional variational autoencoder (VAE). This model was first described in the paper "Neural Discrete Representation Learning", by van der Oord et al., in 2017. 
## Traditional VAEs
A traditional variational autoencoder network consists of three key components: the encoder, a latent space, and a decoder network. The encoder effectively compresses the data to obtain the latent space representation, which is then decoded by the decoder network. An image of a traditional VAE network is shown in ][][][][][image]
## VQ-VAEs
The VQ-VAE differs from a traditional VAE by using a discretised version of the latent space; continuous encoder outputs are mapped (in the latent embedding space) to a series of K embedding vectors (each with dimensionality D). This mapping is done by means of a nearest-neighbour lookup (i.e., using the L2-norm). The vq encoding of the image is the input to the decoder network. 
These embedding vectors themselves are part of the learning process; the vectors are adjusted according to the l2 error terms (i.e., they are moved to more closely fit the data). 
The encoder/decoder networks learn in tandem; the encoder learns to produce optimal encodings for images, and the decoder learns to decode these, to produce accurate reconstructions from the encodings. The structure of a VQ-VAE is displayed in [][][][].

## Dataset - the preprocessed OASIS MRI dataset
The datset utilised was the preprocessed OASIS brain scan dataset supplied on the blackboard site (at this link: https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download). The dataset has been preprocessed, such that all scans are centered and 256*256 pixels. The dataset contains 9664 training images, 544 testing images and 1120 validation images. The images were normalised, by dividing all values by 255.

## The Model
The produced model was based on the open-source solution available in the Keras documentation (at https://keras.io/examples/generative/vq_vae/). Modifications to this code were made, to allow processing of the data on lower-spec hardware (it is able to run on a 6GB RTX3060 Laptop GPU) - this involved adaptation of the code to utilise batching and attempts at implementing mixed-precision. Additional modifications to the structure of the CNN and VQ-VAE were also trialled, including addition of Dropout and BatchNormalisation layers, in an attempt to battle poor generalisability.

## Performance
The VQ-VAE component of the generative task performed well; it was capable of reproducing the brain scans with a SSIM averaging well above 0.8 (typically, closer to 0.95 on the tested images). A series of images, alongside their respective reconstructions, are shown below. 

The PixelCNN generative model did not perform as hoped; the randomly generated encodings did not replicate the structure of the encodings from the brain images, and therefore the decoded images did not closely resemble real brains. Some of the generated brain models are shown below.