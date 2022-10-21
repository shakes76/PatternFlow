import vit_pytorch
import torch
import numpy as np

def build_model(dim=512, 
    image_size=256, 
    patch_size=32,
    num_classes=2,
    depth=6,
    heads=12,
    mlp_dim=1024,
    channels=1,
    dropout=0.3,
    emb_dropout=0.3,
    lr=0.0001):
    """ Initialises ViT model from vit_pytorch and supporting training constructs with supplied parameters an
    
        :param dim          :   Depth of output linear tranform
        :param image_size   :   Size of images
        :param patch_size   :   Size of patches
        :param num_classes  :   Number of classes for classification
        :param depth        :   Number of transformer blocks
        :param heads        :   Number of heads for multi-head attention
        :param mlp_dim      :   Dimension of MLP layer
        :param channels     :   Image chanels (3 for RGB, 1 for grey)
        :param dropout      :   Dropout rate
        :param emb_dropout  :   Embedding dropout rate
        :param lr           :    Learning rate

    Returns:
        model       :   vit_pytorch ViT model
        optimizer   :   pytorch optimizer
        criterion   :   pytorch loss function
        scheduler   :   pytorch learning rate scheduler
    """
    model = vit_pytorch.ViT(
    dim=dim, 
    image_size=image_size, 
    patch_size=patch_size,
    num_classes=num_classes,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    channels=channels,
    dropout=dropout,
    emb_dropout=emb_dropout
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.8)



    return model, optimizer, criterion, scheduler