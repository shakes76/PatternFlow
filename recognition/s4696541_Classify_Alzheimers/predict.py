"""
Example usage of trained alzheimers classification model.
"""
import tensorflow as tf
from modules import AlzheimerModel

assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

if __name__ == "__main__":
    az_model = AlzheimerModel(
        patch_size=16, 
        num_layers=16,
        num_heads=12,
        d_model=64,
        d_mlp=3000,
        head_layers=1000,
        dropout_rate=0.1,
        num_classes=2
    )
    az_model.build((1, 240, 240, 3))

    print(az_model.summary())