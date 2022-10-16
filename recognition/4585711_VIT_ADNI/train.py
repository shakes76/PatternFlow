import tensorflow as tf

from dataset import get_data_preprocessing
from modules import build_ViT
from utils import Params, configure_gpus

if __name__ == "__main__":
    configure_gpus()

    p = Params()

    train_ds, test_ds, preprocessing = get_data_preprocessing(
        image_size=p.image_size(), cropped_image_size=p.cropped_image_size(), cropped_pos=p.cropped_pos(),
        data_dir=p.data_dir())
    model = build_ViT(
        preprocessing=preprocessing, image_size=p.image_size(), transformer_layers=p.transformer_layers(),
        patch_size=p.patch_size(), hidden_size=p.hidden_size(), num_heads=p.num_heads(), mlp_dim=p.mlp_dim(),
        num_classes=p.num_classes(), dropout=p.dropout(), emb_dropout=p.emb_dropout())

    model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

    model.fit(x=train_ds,
          epochs=p.epochs(),
          validation_data=test_ds)

    model.save_weights(p.data_dir() + "checkpoints/my_checkpoint")