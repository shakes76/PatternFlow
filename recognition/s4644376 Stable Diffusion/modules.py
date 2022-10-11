from diffusion_imports import *
from dataset import *

class ConvReLU(nn.Module):
    def __init__(self, num_in, num_out, dimension = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.conv2 = nn.Conv2d(num_out, num_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU()

        self.pos_encoding = nn.Linear(dimension, num_out)

    def forward(self, input_data, pos) -> torch.Tensor:
        #calculate block 1

        out = self.conv1(input_data)
        out = self.bn1(out)
        out = self.relu(out)

        #Calcaulte positon
        pos = self.pos_encoding(pos)
        pos = self.relu(pos)

        # add two dimensions to tensor for addition
        pos = pos[(..., ) + (None, ) * 2]


        out = out + pos

        #calculate block 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UpBlock(nn.Module):

    def __init__(self, num_in, num_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.block = ConvReLU(num_out + num_out, num_out)

    def forward(self, x, skip, pos):
        out = self.up(x)
        out = torch.cat([out, skip], axis=1)

        return self.block.forward(out, pos)

class CalculatePositionEncodingBlock(nn.Module):
    """
    As NN has no sense of time/iteration you encode a POSITION level to the
    amount of noise allowing the nerual network to understand how far away from original
    image

    adapted from:
    https://huggingface.co/blog/annotated-diffusion
    and extended to work seamlessly with the unet built
    """
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.linear = nn.Linear(dimension, dimension)
        self.relu = nn.ReLU()

    def forward(self, time):
        embeddings = math.log(10000) / (self.dimension / 2 - 1)
        embeddings = torch.exp(torch.arange(self.dimension / 2) * - embeddings).cuda()
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        embeddings = self.linear(embeddings)
        return self.relu(embeddings)



class MainNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_dimensions = 32
        # encodes
        self.block1 = ConvReLU(1, 64)
        self.block2 = ConvReLU(64, 128)
        self.block3 = ConvReLU(128, 256)
        self.block4 = ConvReLU(256, 512)

        self.skip_block = ConvReLU(512, 1024)

        # decodes
        self.block5 = UpBlock(1024, 512)
        self.block6 = UpBlock(512, 256)
        self.block7 = UpBlock(256, 128)
        self.block8 = UpBlock(128, 64)

        self.pool = nn.MaxPool2d((2, 2))

        #make tensor 4
        self.final_conv = nn.Conv2d(64, 1, 1)

        #make positional block
        self.pos_block = CalculatePositionEncodingBlock(self.time_dimensions)

    def forward(self, x, position):
        pos = self.pos_block(position)

        out = self.block1(x, pos)
        copy_crop_one = out.clone()
        out = self.pool(out)

        out = self.block2(out, pos)
        copy_crop_two = out.clone()
        out = self.pool(out)

        out = self.block3(out, pos)
        copy_crop_three = out.clone()
        out = self.pool(out)

        out = self.block4(out, pos)
        copy_crop_four = out.clone()
        out = self.pool(out)

        out = self.skip_block(out, pos)

        out = self.block5(out, copy_crop_four, pos)

        out = self.block6(out, copy_crop_three, pos)

        out = self.block7(out, copy_crop_two, pos)

        out = self.block8(out, copy_crop_one, pos)

        out = self.final_conv.forward(out)

        return out






