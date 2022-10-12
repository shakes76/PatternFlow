import tensorflow as tf
import keras
from keras import layers

from vit_tensorflow import ViT

def get_vit(image_size, patch_size=32, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emp_dropout=0.1):
    return ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )