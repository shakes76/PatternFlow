import torch.nn as nn
import torch


def init_weights(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0)
        except AttributeError:
            print(f"Skipping initialisation of {classname}")


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return nn.functional.tanh(x) * nn.functional.sigmoid(y)


class MaskedGatedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, num_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_conditional_embedding = nn.Embedding(num_classes, 2 * dim)

        kernel_shape = (kernel // 2 + 1, kernel)
        padding_shape = (kernel // 2, kernel // 2)
        self.vertical_stack = nn.Conv2d(
            dim, dim * 2, kernel_shape, 1, padding_shape
        )

        self.vertical_to_horizontal = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shape = (1, kernel // 2 + 1)
        padding_shape = (0, kernel // 2)
        self.horizontal_stack = nn.Conv2d(
            dim, dim * 2, kernel_shape, 1, padding_shape
        )

        self.horizontal_residuals = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vertical_stack.weight.data[:, :, -1].zero_()  # mask final row
        self.horizontal_stack.weight.data[:, :, :, -1].zero_()  # mask final column

    def forward(self, x_vertical, x_horizontal, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_conditional_embedding(h)
        h_vertical = self.vertical_stack(x_vertical)
        h_vertical = h_vertical[:, :, :, :x_horizontal.size(-2)]
        out_vertical = self.gate(h_vertical + h[:, :, None, None])

        h_horizontal = self.horizontal_stack(x_horizontal)
        h_horizontal = h_horizontal[:, :, :, :x_horizontal.size(-2)]
        v2h = self.vertical_to_horizontal(h_vertical)

        out = self.gate(v2h + h_horizontal + h[:, :, None, None])
        if self.residual:
            out_horizontal = self.horizontal_residuals(out) + h_horizontal
        else:
            out_horizontal = self.horizontal_residuals(out)

        return out_vertical, out_horizontal


class PixelCNN(nn.Module):
    def __init__(self, input_dimension=256, dim=64, num_layers=15, num_classes=10):
        super().__init__()
        self.dim = dim

        self.embedding = nn.Embedding(input_dimension, dim)

        self.layers = nn.ModuleList

        for i in range(num_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(module=MaskedGatedConv2d(mask_type, dim, kernel, residual, num_classes))

        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dimension, 1)
        )

        self.apply(init_weights)

    def forward(self, x, label):
        shape = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shape)
        x = x.permute(0, 3, 1, 2)

        x_vertical, x_horizontal = (x, x)
        for i, layer in enumerate(self.layers):
            x_vertical, h_horizontal = layer(x_vertical, x_horizontal, label)

        return self.out_conv(x_horizontal)

    def generate(self, label, shape=(8, 8), batch_size = 64):
        param = next(self.parameters())
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=param.device)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = nn.functional.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)

        return x
