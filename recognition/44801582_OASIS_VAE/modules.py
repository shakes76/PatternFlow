"""
Source code of the components of your model. Each component must be implemented as a class or a function
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.LazyConv2d(dim, 3, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(True),
            nn.LazyConv2d(dim, 1),
            nn.LazyBatchNorm2d()
        )

    def forward(self, x):
        return x + self.block(x)


def get_encoder(latent_dim=16):
    enc_model = nn.Sequential(
        nn.LazyConv2d(latent_dim, 4, 2, 1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(True),
        nn.LazyConv2d(latent_dim, 4, 2, 1),
        ResBlock(latent_dim),
        ResBlock(latent_dim),
    )

    return enc_model


class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.vector_quantizer = self.def_vq()
        self.vector_quantizer_straight_through = self.def_vq_straight_through()

    def def_vq(self):
        class Quantization(Function):
            def forward(ctx, inputs, codebook):
                with torch.no_grad():
                    codebook_sqr = torch.sum(codebook ** 2, dim=1)
                    inputs_sqr = torch.sum(inputs.view(-1, codebook.size(1)) ** 2, dim=1, keepdim=True)

                    distances = torch.addmm(codebook_sqr + inputs_sqr,
                                            inputs.view(-1, codebook.size(1)), codebook.t(), alpha=-2.0, beta=1.0)

                    indices = torch.min(distances, dim=1)[1].view(*inputs.size()[:-1])
                    ctx.mark_non_differentiable(indices)

                    return indices

        return Quantization.apply

    def def_vq_straight_through(self):
        class QuantizationST(Function):
            def forward(ctx, inputs, codebook):
                indices = self.vector_quantizer(inputs, codebook).view(-1)
                ctx.save_for_backward(indices, codebook)
                ctx.mark_non_differentiable(indices)

                codes_flatten = torch.index_select(codebook, dim=0,
                                                   index=indices)
                codes = codes_flatten.view_as(inputs)

                return codes, indices

            def backward(ctx, grad_output, grad_indices):
                grad_inputs, grad_codebook = None, None

                if ctx.needs_input_grad[0]:
                    grad_inputs = grad_output.clone()
                if ctx.needs_input_grad[1]:
                    indices, codebook = ctx.saved_tensors

                    grad_codebook = torch.zeros_like(codebook)
                    grad_codebook.index_add_(0, indices, grad_output.contiguous().view(-1, codebook.size(1)))

                return grad_inputs, grad_codebook

        return QuantizationST.apply

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        latent = self.vector_quantizer(x, self.embed.weight)

        return latent

    def straight_through(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_quantized_straight_through, ind = self.vector_quantizer_straight_through(x, self.embed.weight.detach())
        x_quantized_straight_through = x_quantized_straight_through.permute(0, 3, 1, 2).contiguous()

        x_quantized = torch.index_select(self.embed.weight, dim=0, index=ind)\
            .view_as(x).permute(0, 3, 1, 2).contiguous()

        return x_quantized_straight_through, x_quantized


def get_decoder(latent_dim, final_dim):
    dec_model = nn.Sequential(
        ResBlock(latent_dim),
        ResBlock(latent_dim),
        nn.ReLU(True),
        nn.LazyConvTranspose2d(latent_dim, 4, 2, 1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(True),
        nn.LazyConvTranspose2d(final_dim, 4, 2, 1),
        nn.Tanh()
    )

    return dec_model


class VQ_VAE(nn.Module):
    def __init__(self, dim=16, num_embeddings=64):
        super().__init__()
        self.encoder = get_encoder(dim)
        self.codebook = VQ(num_embeddings, dim)
        self.decoder = get_decoder(dim, 1)

    def encode(self, x):
        return self.codebook(self.encoder(x))

    def decode(self, latents):
        return self.decoder(self.codebook.embedding(latents).permute(0, 3, 1, 2))

    def forward(self, x):
        encoding = self.encoder(x)
        x_quantized_straight_through, x_quantized = self.codebook.straight_through(encoding)
        reconstruct = self.decoder(x_quantized_straight_through)
        return reconstruct, encoding, x_quantized
