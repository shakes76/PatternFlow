# %%
from dataset import *
from modules import *
from train import *
from predict import *

# %%
# get required data
train_x = load_images('C:/TechnoCore/2022/COMP3710/project/train_x')
train_y = load_labels('C:/TechnoCore/2022/COMP3710/project/train_y')
valid_x = load_images('C:/TechnoCore/2022/COMP3710/project/valid_x')
valid_y = load_labels('C:/TechnoCore/2022/COMP3710/project/valid_y')

# %%
unet = unet(input_size=(128,128,3), n_filters=32, n_classes=255)
unet.summary()

# %%
unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
             
# %%
results = unet.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(valid_x, valid_y))