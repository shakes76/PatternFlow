# VQ-VAE

# Training
#### VQVAE
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

# Generating
| | Prior | Decoded Images |
| ------------- | ------------- | ------------- |
| Test data | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/test_q.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/test_imgs.png) |
| Generated | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_q.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_imgs.png) |
| Generated | <img src="https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_q2.png" width="600"> | ![](https://github.com/CarrickC/PatternFlow/blob/topic-recognition/recognition/45678044/images/generated_imgs2.png) |

# Usage
~~~text
python vqvae.py -h
usage: vqvae.py [-h] [--epoch EPOCH] [--batch BATCH] [--lr LR] [--k K] [--d D]

VQVAE

optional arguments:
  -h, --help     show this help message and exit
  --epoch EPOCH  Epoch size (default: 50)
  --batch BATCH  Batch size (default: 32)
  --lr LR        learning rate for Adam optimizer (default: 0.002)
  --k K          Num of latent vectors (default: 512)
  --d D          Dim of latent vectors (default: 64)
~~~

# Reference
VQVAE - https://arxiv.org/pdf/1711.00937.pdf  
PixelCNN - https://arxiv.org/pdf/1606.05328.pdf
