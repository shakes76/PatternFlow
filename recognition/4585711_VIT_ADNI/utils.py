from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

class Params():
    def __init__(self, file="config.yaml"):
        with open(file, 'r') as f:
            self.yaml = load(f, Loader)
    
    def data_dir(self): return self.yaml["data_dir"]
    def image_size(self): return tuple(self.yaml["image_size"])
    def cropped_image_size(self): return tuple(self.yaml["cropped_image_size"])
    def cropped_pos(self): return tuple(self.yaml["cropped_pos"])
    def batch_size(self): return self.yaml["batch_size"]
    def transformer_layers(self): return self.yaml["transformer_layers"]
    def patch_size(self): return self.yaml["patch_size"]
    def hidden_size(self): return self.yaml["hidden_size"]
    def num_heads(self): return self.yaml["num_heads"]
    def mlp_dim(self): return self.yaml["mlp_dim"]
    def num_classes(self): return self.yaml["num_classes"]
    def dropout(self): return self.yaml["dropout"]
    def emb_dropout(self): return self.yaml["emb_dropout"]