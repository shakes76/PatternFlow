# %%
from modules import *
from dataset import *

# %%
# get required data
train_x = load_images('C:/TechnoCore/2022/COMP3710/project/train_x')
train_y = load_labels('C:/TechnoCore/2022/COMP3710/project/train_y')
valid_x = load_images('C:/TechnoCore/2022/COMP3710/project/valid_x')
valid_y = load_labels('C:/TechnoCore/2022/COMP3710/project/valid_y')

# %%
def train_unet(train_x, train_y, valid_x, valid_y):
    """
    Trains the model on the training data and returns results

    :return: results of training
    """

    unet = unet_full(input_size=(128,128,3), n_filters=32)
    unet.summary()
    unet.compile(optimizer=tf.keras.optimizers.Adam(), 
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    results = unet.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(valid_x, valid_y))
    return unet, results

# %%
# train the model
(unet, results) = train_unet(train_x, train_y, valid_x, valid_y)

# %%
# saves the model for future use
tf.keras.models.save_model(unet, 'C:/TechnoCore/2022/COMP3710/project_upload/PatternFlow/dice_problem')
