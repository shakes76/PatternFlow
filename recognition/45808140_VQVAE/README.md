# VQVAE on ADNI dataset for generating brain images

**Description of algortihm and problem it solves 1 paragraph**

**figure and visualisations**


**dependencies versions and reproducibility of results**


**example inputs outputs and plots of algorithm**


**pre-processing with references justify train,validate and test splits**



encoder gives discrete codes rather than conts, priors are learnt rather than being static

latent vector is hidden

encoder outputs means and log(stds) 

minimise recons loss

q(z|x)

miinimise the posterior given the prior

regularising latent space

find closest L2 norm codebook

discrete latent codebook vectors that are learnable, decoder we take out the 

forward propagate is standard, back propagate, gradients from decoder to encoder then encoder knows how to change information to lower recons loss 

encoder optimises both first and last loss terms, 

training vqvae prior is kept constant and uniform and then we fit autoregressive distribution to generate x using ancestral sampling