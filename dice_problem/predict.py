# %%
from modules import *
from dataset import *

from keras import backend as k
from matplotlib import pyplot as plt
import tensorflow as tf

# %%
unet = tf.keras.models.load_model('C:/TechnoCore/2022/COMP3710/project_upload/PatternFlow/dice_problem')

# %%
def predict(model, image):
    """
    Used to predict a single segmentation.

    :return: a single segmentation prediction
    """
    pred = model.predict(image)
    return pred

# %%
def dice_coefficient(model, valid_x, valid_y):
    """
    Calculates the dice coefficient

    :return: dice coefficient
    """
    pred_y = []
    for image in valid_x:
        pred_y.append(model.predict(image))
    
    true_y_f = k.flatten(valid_y)
    pred_y_f = k.flatten(pred_y) 
    
    intersection1 = k.sum(true_y_f * pred_y_f)

    dice = (2.0*intersection1) / (k.sum(true_y_f) + k.sum(pred_y_f))

    return dice  

# %%
# predict a variable
img_name = glob.glob('C:/TechnoCore/2022/COMP3710/project/test_x'+'/*.jpg')[0]
img = Image.open(img_name).convert('RGB')
img = img.resize((128, 128))
img = np.reshape(img, (128, 128, 3)) 
img = img / 256

label_name = glob.glob('C:/TechnoCore/2022/COMP3710/project/test_y'+'/*.png')[0]
label = Image.open(label_name).convert('RGB')
label = label.resize((128, 128))
label = np.reshape(label, (128, 128, 3)) 

pred = predict(unet, img[np.newaxis, ...])

# %%
plt.imshow(img)
# %%
plt.imshow(label)
# %%
plt.imshow(pred[0])
# %%
