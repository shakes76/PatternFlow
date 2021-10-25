This is my first style GAN impelementation. Because of short in GPU, the model is intermittently trained on Colab.

# How the mapping network dis-entangle the random tensor?
## Search for distribution
We first generate 4 sets of random vectors with the dimension of (2000,512) by using 3 different ways, torch.randn(Normal Distribution), torch.rand(Uniform Distribution), and torch.randint(Uniform Distribution with integers). Then, we compared the latent code z (z1,z2,z3,z4) before going into the mapping network with the intermediate latent code w (w1,w2,w3,w4) about their means and standard deviation as the table below:

```python
#The data typr transformation will not be provided in this script, just basic ideas
z1 = torch.randn(2000, 512).to(device)
z2 = torch.rand(2000, 512).to(device)
z3 = torch.randint(200,400,(2000, 512),dtype=torch.float32).to(device)
z4 = torch.randint(10000,80000,(2000, 512),dtype=torch.float32).to(device)

w1,w2,w3,w4 = mapNet(z1), mapNet(z2), mapNet(z3), mapNet(z4)
```

| W  | W-mean | W-std |
| -- | ---- | --- |
| w1 | 0.0043377355  | 0.026536208  |
| w2 | 0.0044174353  | 0.027271809  |
| w3 | 0.0045437375  | 0.02691137   |
| w4 | 0.004477056   | 0.027103048  |

| Z  | Z-mean | Z-std |
| -- | ---- | --- |
| z1 | 9.3621886e-05  | 0.99944264  |
| z2 | 0.5001404  | 0.28848535  |
| z3 | 299.55765  | 57.77888   |
| z4 | 45018.316   | 20201.037  |

**We learned that one of the Mapping Network's function is to transform the random vector to a particular distribution with fixed mean and standard deviation.**
We also made two plots
![alt-text-1](https://github.com/Wangxinqian/PatternFlow/blob/413216e5e6a31c9ebf87b7cc1f87f8f0fe0860b8/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/w_mean_std.png) ![alt-text-2](https://github.com/Wangxinqian/PatternFlow/blob/413216e5e6a31c9ebf87b7cc1f87f8f0fe0860b8/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/z_mean_std.png)
## Search for connection between different columns
