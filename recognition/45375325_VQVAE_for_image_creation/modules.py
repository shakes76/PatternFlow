import torch
import torch.nn as nn
import numpy as np
"""
This file contains all of the models used for the VQVAE. 
"""
DEVICE = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")


class ResidualLayer(nn.Module):
    """
    One residual layer
    """
    def __init__(self, input_dim, latent_dim, residuals_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim, residuals_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(residuals_dim, latent_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x += self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers
    """
    def __init__(self, input_dim, latent_dim, residuals_dim, num_stacked_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = num_stacked_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(input_dim, latent_dim, residuals_dim)] * num_stacked_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = nn.functional.relu(x)
        return x


class Quantizer(nn.Module):
    """
    Discretizer
    """
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(Quantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings / self.num_embeddings)

    def forward(self, z):
        """
        transforms the encoder network z to a discrete one-hot vector mapping that is the index
        of the closest embedding vector e_j
        :param z: the encoder network to be quantized
        :return: loss, quantized z z_q, perplexity, minimum encodings, minimum encoding indicies
        """
        # convert z from z.shape = (batch, channel, height, width) to (batch, height, width, channel)
        z = z.permute(0, 2, 3, 1).contiguous()
        # then flatten
        z_flattened = z.view(-1, self.embedding_dim)

        # z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # calculate closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings
        ).to(DEVICE)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute embedding loss
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # maintain gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        mean_embeddings = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(mean_embeddings + torch.log(mean_embeddings + 1e-10)))

        # reshape quantized z to look like original input
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class Encoder(nn.Module):
    """
    Encoding network

    given data x, maps the data to latent space
    """
    def __init__(self, input_dim, latent_dim, num_residual_layers, num_stacked_layers):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_steps = nn.Sequential(
            nn.Conv2d(input_dim, latent_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(latent_dim, latent_dim, num_residual_layers, num_stacked_layers)
        )

    def forward(self, x):
        return self.conv_steps(x)


class Decoder(nn.Module):
    """
    Decoder network

    given a latent sample z, maps to the original space
    """
    def __init__(self, input_dim, latent_dim, num_residual_layers, num_stacked_layers):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_steps = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim, latent_dim, kernel_size=kernel-1, stride=stride-1, padding=1
            ),
            ResidualStack(latent_dim, latent_dim, num_stacked_layers, num_residual_layers),
            nn.ConvTranspose2d(
                latent_dim, latent_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_steps(x)


class VQVAE(nn.Module):
    def __init__(self, latent_dim, res_h_dim, num_residual_layers,
                 num_embeddings, embedding_dim, beta, save_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image
        self.encoder = Encoder(3, latent_dim, num_residual_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(latent_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent dim to quantizer
        self.vector_quantizer = Quantizer(num_embeddings, embedding_dim, beta)
        # decode discrete latent repr
        self.decoder = Decoder(embedding_dim, latent_dim, num_residual_layers, res_h_dim)

        if save_embedding_map:
            self.embedding_map = {i: [] for i in range(num_embeddings)}
        else:
            self.embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print(f"Original data shape: {x.shape}")
            print(f"Encoded data shape: {z_e.shape}")
            print(f"Reconstructed data shape: {x_hat.shape}")
            assert False

        return embedding_loss, x_hat, perplexity


if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>> initialise test data >>>>>>>>>>>>>>>>>>>
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    # <<<<<<<<<<<<<<<<<<< initialise test data <<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>> test residual classes >>>>>>>>>>>>>>>>>>>>>
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
    # <<<<<<<<<<<<<<<<< test residual classes <<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>> test encoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    # <<<<<<<<<<<<<<<<< test encoder <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>> test decoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)

    # <<<<<<<<<<<<<<<<< test decoder <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

