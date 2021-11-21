# Generative Model of the OASIS Brain using VQ-VAE and PixelCNN
  
The Vector Quantised Variational AutoEncoder (VQ-VAE), a powerful generative model is used to generate reconstructed images of the OASIS brain MRI images. The VQ-VAE differs from the traditional VAE, since the encoder network outputs discrete rather than continuous codes and the prior is learnt rather than static. Vector quantisation (VQ) is used to learn these discrete latent representations. The PixelCNN, an auto-regressive generative model is used to train these codes to generate novel examples from the prior.

Additionally, the images constructed by the VQVAE model should achieve a Structured Similarity (SSIM) of over 0.6 for this task.

## OASIS Brain Dataset
For the current task, the pre-processed OASIS MRI dataset has been obtained. The VQVAE model uses 9,664 training images and 544 test images from the respective folders. There are also 1,120 validation images that are not used in the current implementation. Note, the additional brain segment images are ignored for this task. An example visualisation of the dataset is shown below.

![image](https://user-images.githubusercontent.com/55978813/139571071-3daa6362-09a1-4cd0-8360-e215d2b8f33c.png)
 
## Data Pre-processing
To pre-process the data, the pixel values of the images are normalised to be between [-0.5, 0.5]. Note, in the current implementation, images have been downsampled to 80x80 to account for limited memory resources. Additionally, the variance of the whole training set is calculated. 

## VQVAE Architecture
The VQ-VAE works on a discrete latent space by maintaining a discrete codebook. The encoder models a categorical distribution. The codebook is constructed by discretising the distance between continuous embeddings and the encoded outputs. These discrete code words are passed into the decoder. The decoder is then trained to generate reconstructed samples.

![image](https://user-images.githubusercontent.com/55978813/139571760-e9bc2c2b-ca3e-4d79-b397-9e894ef5535a.png)
*Figure 1: Left: A figure describing the VQ-VAE. Right: Visualisation of the embedding space. The
output of the encoder z(x) is mapped to the nearest point e2. The gradient ∇zL (in red) will push the
encoder to change its output, which could alter the configuration in the next forward pass.*

## PixelCNN Architecture

The PixelCNN model is used to train the discrete codes so they can be used as priors for novel image generation. Generation is done iteratively, and the probability distribution of prior elements determines the probability distribution of later elements. The PixelCNN generates an image pixel by pixel via a masked convolution kernel. This convolution kernel uses data from previously generated pixels (origin at the top left as shown below) to generate the next pixels.

![image](https://user-images.githubusercontent.com/55978813/139572034-6ee9207a-a374-4155-b42f-a707fca75c5b.png)    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   ![image](https://user-images.githubusercontent.com/55978813/139572358-f5508f69-3359-4bd6-9b1a-9e55b4b8fc64.png)


*Figure 2: A visualization of the PixelCNN that &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                              Figure 3: An example matrix that is used to mask <br />                                            maps a neighborhood of pixels to prediction for          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                    the 5x5 filters to make sure the model  cannot <br />
the next pixel. To generate pixel xi the model can            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                cannot read pixels below (or strictly to the right)<br />                                             only condition on the previously generated         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                             of the current pixel to make its predictions.<br />
pixels x1, . . . xi−1.*                       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   <br />

## Training
#### VQVAE
The pre-processed data set contains a 0.85/0.1/0.05 split of training, testing and validation data. A large training set is required so that the model has sufficient information, can learn from a variety of data and produce accurate reconstructions of images. The testing set is later used to visualise reconstructions and evaluate the model. It is essential that the model has not seen the test set before. The validation set is not required, since the parameter tuning is performed by judging the quality of reconstructions produced on the test set. <br />
The VQVAE model is trained to 30 epochs with a batch size of 128.  <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573217-5832c635-3909-4d85-9b44-036cd8684804.png)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Figure 3: VQVAE Reconstruction Loss*

#### PixelCNN
The codebook indices are the inputs to the PixelCNN model which is trained to 50 epochs with a batch size of 128 and a validation split of 0.1. <br />Note that, 90% of the data is used for training so that the model can learn from a large variety of samples and produce accurate results, whereas 10% is used for validation to verify the accuracy of the model and tune the parameters accordingly.  <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573236-fdf37e4c-82f6-4523-a4a3-440f40f3c945.png)  &nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573243-07233a5f-ca49-49f5-9147-1b1370566dfb.png)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 4: Training loss vs. Validation loss curve.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 5: Training accuracy vs. Validation accuracy curve.

## Results
**The reconstructed test images achieved a mean Structured Similarity (SSIM) of 0.82.** <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573406-eed3377e-e526-4931-8c4b-0380a9458b23.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573408-09459b77-1a3e-45dc-b80f-55793a31af1c.png) <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573520-c4487320-8420-4425-a67a-32021d951915.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573415-edc7903b-4b47-4f04-9f32-978e13a7c863.png) <br /><br /><br />

 **Discrete encodings were generated from the original images.** <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573734-fd3ec5af-ffba-471f-be8e-16f22c61eeb8.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573735-b07068ea-5182-45ac-b6c1-4adb3005e624.png) <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573745-3488b659-fe5c-4f04-9c08-3bc1c65b5296.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573749-aed7f213-68f6-4d9d-879d-9e3d55134047.png)
<br /><br /><br />

 **The novel images generated by the discrete encodings have reasonable clarity.** <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573896-ee7c05e2-d687-42b9-a329-a4edde9d7de3.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573903-4fbd364b-673f-45eb-ba26-823e212d9b1c.png) <br /><br />
![image](https://user-images.githubusercontent.com/55978813/139573907-e48920e3-68c9-4278-b3dc-a63b2629ef88.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/55978813/139573924-6521d638-c884-4262-af16-49ac56f96e9e.png)
<br />

## Dependencies 
•	Python 3.7 <br />
•	TensorFlow 2.6.0 <br />
•	Numpy 1.19.5 <br />
•	matplotlib 3.2.2 <br />
•	Pillow 7.1.2 <br />
•	tensorflow-probability 0.14.1 <br />
•	os  <br />
•	Pre-processed OASIS MRI dataset (accessible at <https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA>).

## References
[1] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, 2018. Neural Discrete Representation Learning. [Online]. Available at: <https://arxiv.org/pdf/1711.00937.pdf>. <br />
[2] Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves, A. and Kavukcuoglu, K., 2016. Conditional Image Generation with PixelCNN Decoders. [online] Available at: <https://arxiv.org/pdf/1606.05328.pdf>. <br />
[3] Paul, S., 2021. Keras documentation: Vector-Quantized Variational Autoencoders. [online] Keras.io. Available at: <https://keras.io/examples/generative/vq_vae/>.



