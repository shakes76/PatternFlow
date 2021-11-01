# Perceiver

Note that I unfortunately couldn't acquire the ADNI dataset in time, hence I've used MNIST as a replacement.

The Perceiver [1] model is a general architecture based on attentional principles designed to handle arbitrary configurations of various modalities using 
a single Transformer-based architecture. However unlike generic transformer architectures, the Perceiver first maps inputs to a small latent space which makes it possible to build complex and deep networks regardless of input size.

![image](https://user-images.githubusercontent.com/26598965/139605460-35962644-ae3c-4629-8b0f-19c39002f8d0.png)

### Architecture
The Perceiver consists of two main components: The cross attention and the latent transformer (self attention). The perceiver first uses a cross attention layer to project a high-dimensional byte array to a latent bottleneck, prior to processing by self-attention blocks in the latent space. This process is iterated alternatively.

## Dependencies
- Pytorch 1.9.0
- Python >= 3.6
- tqdm (for notebook visualization)
- Matplotlib

## Usage
```
model = Perceiver(
  depth=6, <- The amount of blocks and depth of the network.
  num_channels=1, <- Number of channels in input (eg: color channels)
  input_shape=1, <- Input dimensions
  fourier_bands=4, <- Amount of bands for positional encoding 
  num_latents=16, <- Number of latent vectors
  latent_dim=64, <- Dimensionality of latent vectors.
  latent_heads=8, <- Amount of heads in self-attention block
  attn_dropout=0., <- Dropout probability
  num_features=10, <- Number of output features
  self_per_cross_atn=4, <- The number of self-attention layers per cross attention layer
)
```

## Examples
Accuracy Plot:
![image](https://user-images.githubusercontent.com/26598965/139607812-6365495c-9bd8-4bfc-8e33-762f16e2af7c.png)

## References
[1] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals,
João Carreira.
*Perceiver: General Perception with Iterative Attention*. ICML 2021.
https://arxiv.org/abs/2103.03206

