import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from perceiver_pytorch import PerceiverIO
from dataset import ADNIDataset

class ADNIClassifier(nn.Module):
    RESNET_FEATURES = 2048

    def __init__(self):
        super().__init__()
        self.pre_resnet = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*tuple(resnet.children())[:-1])
        self.perceiver = PerceiverIO(
            dim = 2048,                  # dimension of sequence to be encoded
            queries_dim = 1280,          # dimension of decoder queries
            logits_dim = 1,              # dimension of final logits
            depth = 6,                   # depth of net
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 1280,           # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
        )
        self._queries = nn.Parameter(0.02 * torch.randn(1, 1280))

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.reshape((B * S, C, H, W))
        out = self.pre_resnet(x)
        out = self.resnet(out)
        out = out.reshape((B, S, ADNIClassifier.RESNET_FEATURES))
        out = self.perceiver(out, queries=self._queries)
        out = torch.sigmoid(out)
        return out

if __name__ == "__main__":
    device = torch.device("cpu")
    ds = ADNIDataset("/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC", device)
    img_seq, label = ds[0]
    img_seq = img_seq.unsqueeze(0)
    print(img_seq.shape)
    print(label.shape)
    queries = torch.randn(256, 1280)

    img_seq = torch.rand((4, 20, 1, 240, 256))
    model = ADNIClassifier().to(device)
    out = model(img_seq)
    print(out.shape)
