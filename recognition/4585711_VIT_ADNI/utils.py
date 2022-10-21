from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import tensorflow as tf

from dataset import get_data_preprocessing
from modules import build_ViT

class Params():
    def __init__(self, file="config.yaml"):
        with open(file, 'r') as f:
            self.yaml = load(f, Loader)
    
    def data_dir(self): return self.yaml["data_dir"]
    def image_dir(self): return self.yaml["image_dir"]
    def image_size(self): return tuple(self.yaml["image_size"])
    def cropped_image_size(self): return tuple(self.yaml["cropped_image_size"])
    def batch_size(self): return self.yaml["batch_size"]
    def transformer_layers(self): return self.yaml["transformer_layers"]
    def patch_size(self): return self.yaml["patch_size"]
    def hidden_size(self): return self.yaml["hidden_size"]
    def num_heads(self): return self.yaml["num_heads"]
    def mlp_dim(self): return self.yaml["mlp_dim"]
    def num_classes(self): return self.yaml["num_classes"]
    def dropout(self): return self.yaml["dropout"]
    def emb_dropout(self): return self.yaml["emb_dropout"]
    def epochs(self): return self.yaml["epochs"]
    def learning_rate(self): return self.yaml["learning_rate"]

def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def init_model():
    configure_gpus()

    p = Params()

    train_ds, test_ds, valid_ds, preprocessing = get_data_preprocessing(
        batch_size=p.batch_size(), image_size=p.image_size(), cropped_image_size=p.cropped_image_size(),
        data_dir=p.data_dir())
    model = build_ViT(
        preprocessing=preprocessing, image_size=p.image_size(), transformer_layers=p.transformer_layers(),
        patch_size=p.patch_size(), hidden_size=p.hidden_size(), num_heads=p.num_heads(), mlp_dim=p.mlp_dim(),
        num_classes=p.num_classes(), dropout=p.dropout(), emb_dropout=p.emb_dropout())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate()), 
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

    return train_ds, test_ds, valid_ds, preprocessing, model, p