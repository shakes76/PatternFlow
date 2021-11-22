# Generative Model of the OASIS Brain using VQ-VAE

## VQ-VAE
<p align="center">
  <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/vqvae.PNG" width="800">
</p>  
<p>
VQVAE extends the autoencoder by maintaining an additional codebook which will be used for quantizing the continuous latent code into discrete vector representation. The output from the encoder will be compared to all the categorical features vectors in the codebook and a table of codebook indices will be computed based on the closest Euclidean distance. Then, the corresponding categorical features vector will be used to replace the original ones, which is called "Vector Quantized".  
</p>
<p>
The main purpose of training is to allow the codebook to learn the underlying categorical features or rather the discrete latent space contained by the image set. After training VQVAE, we could use the encoder to train a PixelCNN model on the same image set in order to learn the piror distribution between images and categorical features distribution. Then we could use the PixelCNN model to produce the table of codebook indices based on the piror distribution and get the discrete vector representation with the trained codebook to achieve the ability of generating. 
</p>


## Training
#### VQVAE
The OASIS MRI dataset contains 9,664 images for training, 1,120 images for validation and 544 images for testing. After 20 epochs, both training and validation ssim converge to around 0.93. The testing ssim is evaulated to be 0.9  
```text
Average SSIM on test dataset: 0.906753911253284
```
<p align="center">
  <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/vqvae_train_status.png" width="600">
</p>  


| | |
| ------------- | ------------- |
| Original | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/original.png) |
| Reconstructed | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/reconst.png) |  


#### PixelCNN Prior

<p align="center">
  <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/prior_train_status.png" width="600">
</p>  

## Generating
| | q(z/x) | Decoded Images |
| ------------- | ------------- | ------------- |
| Test data | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/test_q.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/test_imgs.png) |
| Generated | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_q.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_imgs.png) |
| Generated | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_q2.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_imgs2.png) |  

## Usage
~~~text
python vqvae.py -h
usage: vqvae.py [-h] [--epoch EPOCH] [--batch BATCH] [--lr LR] [--k K] [--d D]

VQVAE

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH         Epoch size for training vqvae (default: 50)
  --batch BATCH         Batch size for training vqvae (default: 32)
  --lr LR               learning rate for training vqvae (default: 0.002)
  --epoch_prior EPOCH_PRIOR
                        Epoch size for training pixelcnn (default: 100)
  --batch_prior BATCH_PRIOR
                        Batch size for training pixelcnn (default: 64)
  --lr_prior LR_PRIOR   learning rate for training pixelcnn (default: 0.001)
  --k K                 Num of latent vectors (default: 512)
  --d D                 Dim of latent vectors (default: 64)
~~~  

## Requirements
python 3.6.9  
torch 1.9.0+cu111  
numpy 1.19.5  
matplotlib 3.2.2  
natsort 7.1.1  
tqdm 4.62.3  
PIL 8.3.1  
argparse 1.1  

## Reference
VQVAE - https://arxiv.org/pdf/1711.00937.pdf  
PixelCNN - https://arxiv.org/pdf/1606.05328.pdf
