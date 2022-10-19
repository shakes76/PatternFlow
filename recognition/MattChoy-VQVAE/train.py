from constants import latent_dimensions, batch_size, num_embeddings
from modules import Encoder, Decoder, VectorQuantiser

# Test Encoder and Decoder sub-components
enc = Encoder(latent_dimensions)
dec = Decoder()

enc.build(input_shape=(batch_size, 256, 256, 3))
dec.build(input_shape=enc.out_shape)
print(enc.summary())
print(dec.summary())

# Test VQ subcomponent
vq = VectorQuantiser(num_embeddings, latent_dimensions)
vq.build(input_shape=enc.out_shape)
