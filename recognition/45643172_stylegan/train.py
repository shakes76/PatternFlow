import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from glob import glob
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import InstanceNormalization

from dataset import *
from modules import *

"""We first build the StyleGAN at smallest resolution, 
such as 4x4 or 8x8. Then we progressively grow the model 
to higher resolution by appending new generator and discriminator blocks."""
START_RES = 4
TARGET_RES = 16

def load_model():
    
    return StyleGAN(start_res=START_RES, target_res=TARGET_RES)


#The training for each new resolution happen in two phases - "transition" and "stable". 

def train(
    start_res=START_RES,
    target_res=TARGET_RES,
    steps_per_epoch=5000,
    display_images=True,
):

    style_gan = StyleGAN(start_res=4, target_res=128)
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    v_batch_size = 16
    v_z = tf.random.normal((v_batch_size, style_gan.z_dim))
    v_noise = style_gan.generate_noise(v_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dataloader = create_dataloader(res)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            ckpt_cb = keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                train_dataloader, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )

            #In the transition phase, the features from the previous resolution are mixed with the current resolution. 
            #This allows for a smoother transition when scalling up. We use each epoch in model.fit() as a phase.

            if display_images:
                images = style_gan({"z": v_z, "noise": v_noise, "alpha": 1.0})
                plot_images(images, res_log2)
                
    # plot the losses of discriminator and generator
    plt.plot(style_gan.all_d_loss)
    plt.plot(style_gan.all_g_loss)
    plt.title('loss of training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['dis_loss', 'gen_loss'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    train(steps_per_epoch=10000)

