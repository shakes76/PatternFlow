Brief introduction on the perceiver transformer architecture

Perceiver transformer architecture:

Introduction: Usual transformers have the problem of quadratic bottle-neck(output query vector has too much of a dimension to handle). In the perceiver transformer architecture, high dimensional input such as images, video is represented as byte array, where it uses cross-attention with the use of Latent array to solve this problem (The input sequence gets transformed to a shorter sequence through the Latent state). 

To be more specific, in each layer of the perceiver transformer architecture, the inputs are fed into the each of the cross attention-module with the use of the Latent array to perform cross-attention. This allows for a deeper neural net when compared to usual transformer model. 
Moreover, in the perceiver transformer architecture, the transformers are invariant to positional encoding of the data, i.e. different permutation of the input data does not affect the overall model. This is accomplished through the Fourier encoding as described in the paper. 

As such, the perceiver transformer model consist of the following:
-	Cross attention module (to reduce dimension of input size)
-	Latent transformer layer after applying cross attention with the input byte array (usual transformer layer for handling data after cross attention)
-	Fourier encoding (for invariance to positional encoding of data)

