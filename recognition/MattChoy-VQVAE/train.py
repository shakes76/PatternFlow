from constants import latent_dimensions, batch_size
from modules import Encoder, Decoder

enc = Encoder(latent_dimensions)
dec = Decoder()

enc.build(input_shape=(batch_size, 256, 256, 3))
dec.build(input_shape=enc.out_shape)
print(enc.summary())
print(dec.summary())
