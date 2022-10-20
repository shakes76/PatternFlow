from dataset import *
from modules import *

EPOCHS = 5


def init_encoder_and_quantizer(model):
    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")
    return encoder, quantizer

def flatten_outputs(train_data, encoder):
    # flatten the encoder outputs
    encoded_outputs = encoder.predict(train_data)
    # reduce indices because my VRAM is insufficient
    encoded_outputs = encoded_outputs[:len(encoded_outputs) // 2]
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    return encoded_outputs, flat_enc_outputs

def init_train_vqvae():
    # initialise and train
    train_images, test_images, train_data, test_data, data_variance = load_dataset()

    vqvae = Train_VQVAE(data_variance, dim=16, embed_n=128)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae_hist = vqvae.fit(train_data, epochs=EPOCHS, batch_size=128)

    vqvae_model = vqvae.vqvae
    vqvae_model.save("vqvae.h5")

    enc, quant = init_encoder_and_quantizer(vqvae_model)
    
    return flatten_outputs(train_data, enc) + (quant, vqvae)

def init_train_pcnn(encoded_outputs, flat_enc_outputs, quant, vqvae):
    # generate the codebook indices
    codebook_indices = quant.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    pcnn = get_pixelcnn(vqvae, encoded_outputs)

    # compile PCNN model
    pcnn.compile(optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],)

    # train PCNN model
    pcnn_hist = pcnn.fit(x=codebook_indices, y=codebook_indices, 
                    batch_size=128, epochs=EPOCHS, validation_split=0.2,)

    pcnn.save("pcnn.h5")

def main():
    encoded_outputs, flat_enc_outputs, quant, vqvae = init_train_vqvae()
    init_train_pcnn(encoded_outputs, flat_enc_outputs, quant, vqvae)

if __name__ == "__main__":
	main()
