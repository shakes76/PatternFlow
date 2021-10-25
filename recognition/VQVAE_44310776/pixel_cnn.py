import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d


def gate(p):
    """
    This is the blue diamond in Figure 2.
    """
    p1, p2 = p.chunk(2, 1)
    return torch.tanh(p1) * torch.sigmoid(p2)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, mask_type='b', residual=True):
        super().__init__()
        assert kernel_size % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.vert_stack = nn.Conv2d(dim,
                                    dim * 2,
                                    kernel_size=(kernel_size // 2 + 1, kernel_size),
                                    stride=1,
                                    padding=(kernel_size // 2, kernel_size // 2)
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        self.horiz_stack = nn.Conv2d(dim,
                                     dim * 2,
                                     kernel_size=(1, kernel_size // 2 + 1),
                                     stride=1,
                                     padding=(0, kernel_size // 2)
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, condition):
        if condition is None:
            condition = 0

        if self.mask_type == 'a':
            self.make_causal()

        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :] #TODO work out what this is doing...
        out_v = gate(h_vert + condition)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = gate(v2h + h_horiz + condition)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class PixelCNN(nn.Module):
    def __init__(self, codebook_size, feature_channels, n_layers):
        super().__init__()

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(codebook_size, feature_channels)
        
        # Initial block with Mask-A convolution
        self.input_conv = GatedMaskedConv2d(feature_channels, kernel_size=7, mask_type='a', residual=False)

        self.layers = nn.ModuleList([
            GatedMaskedConv2d(feature_channels)
            for _ in range(n_layers)])

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(feature_channels, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, codebook_size, 1)
        )
        
        self.condition_embedding = nn.Embedding(codebook_size, feature_channels)
        self.condition_transpose = nn.ConvTranspose2d(feature_channels, feature_channels*2, kernel_size=2, stride=2, padding=0)

    def forward(self, x, condition=None):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, H)

        if condition is not None:
            shp = condition.size() + (-1, )
            condition = self.condition_embedding(condition.view(-1)).view(shp) # (B, H, W, C)
            condition = condition.permute(0, 3, 1, 2).contiguous() # (B, C, W, H)
            condition = self.condition_transpose(condition)


        x_v, x_h = (x, x)
        for _, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, condition)

        return self.output_conv(x_h)

    def generate(self, shape, batch_size, condition=None):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, condition)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x