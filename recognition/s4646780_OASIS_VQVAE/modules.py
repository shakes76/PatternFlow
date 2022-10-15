import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompositeBlockEncoder, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=2))

    def forward(self, input):
        return self.block(input)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = CompositeBlockEncoder(in_channels, num_hiddens)
        self._conv_2 = CompositeBlockEncoder(num_hiddens, num_hiddens // 2)
        # self._conv_3 = CompositeBlockEncoder(num_hiddens//2, num_hiddens//2)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        # x = self._conv_3(x)
        return x


class CompositeBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompositeBlockDecoder, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.ReLU(True),
                                   nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, input):
        return self.block(input)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Decoder, self).__init__()
        # self._conv_1 = CompositeBlockDecoder(in_channels, num_hiddens)
        self._conv_2 = CompositeBlockDecoder(in_channels, num_hiddens)
        self._conv_3 = CompositeBlockDecoder(num_hiddens, num_hiddens * 2)
        self._conv_4 = nn.Conv2d(num_hiddens * 2, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, inputs):
        # x = self._conv_1(inputs)
        x = self._conv_2(inputs)
        x = self._conv_3(x)
        x = self._conv_4(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, torch.argmin(distances, dim=1)

    def get_quantized(self, x):
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(1, 64, 64, 64)
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    def __init__(self,  num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(1, 16)
        self._pre_vq_conv = nn.Conv2d(in_channels=8,
                                      out_channels=embedding_dim,kernel_size=1,
                                      stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, 8)


    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, d = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity