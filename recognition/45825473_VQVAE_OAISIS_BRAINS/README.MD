# Implementation of generative VQVAE for preprocessed OASIS

## The VQVAE

    //To be added

## Model architecture

    //To be added
    ![image]

## Preprocessed OASIS brain dataset

    //To be added

## Results

![image]

## Dependencies

The list of dependencies required for this implementation are as follows:

- python 3
- tensorflow 2.6.0
- tensorflow-probability 0.14.0
- numpy 1.21.3
- matplotlib 3.4.3

## Usage

!ENSURE YOU HAVE THE Data slices folder located in the same folder as 45825473_VQVAE_OASIS_BRAINS

To train the model based on the VQVAE model architexture in

```bash
$ python3 train.py
```

```bash
$ python3 predict.py [-m <PathToPreTrainedModel>]
```

## Example outputs & plots

//To be added

## References

[1] S. Paul, "Vector-Quantized Variational Autoencoders", _keras.io_, Jul. 21, 2021. [Online]. Available: https://keras.io/examples/generative/vq_vae/ [Accessed: Oct. 16, 2022]

[2] ADMoreau, "PixelCNN", _keras.io_, May 26, 2020. [Online]. Available: https://keras.io/examples/generative/pixelcnn/ [Accessed: Oct. 16, 2022]

[3] Sieun Park "An overview on VQ-VAE: Learning Discrete Representation Space" Apr. 4, 2021. Available: https://miro.medium.com/max/720/1*miNfFc9qT5PrS7ectJa_kw.png [Accessed: Oct. 17, 2022]
