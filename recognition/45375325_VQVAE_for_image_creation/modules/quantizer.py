import torch
import torch.nn as nn

DEVICE = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")


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
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

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
        d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
             + torch.sum(self.embedding.weight ** 2, dim=1)
             - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        # calculate closest encodings
        train_indices = torch.argmin(d, dim=1)
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=DEVICE)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute embedding loss
        embedding_loss = nn.functional.mse_loss(z_q.detach(), z)
        q_latent_loss = nn.functional.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.beta * embedding_loss

        # maintain gradients
        z_q = z + (z_q - z).detach()
        # reshape quantized z to look like original input

        return loss, z_q.permute(0, 3, 1, 2).contiguous(), min_encodings, train_indices

    def get_quantized(self, x):
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=DEVICE)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(1, 64, 64, 64)
        return quantized.permute(0, 3, 1, 2).contiguous()
