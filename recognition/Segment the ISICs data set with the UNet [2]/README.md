# README

>  This is the problem 3: *Segment the ISICs data set with the UNet [2] with all labels having a minimum Dice similarity coefficient of 0.7 on the test set. [Easy Difficulty]*  



### Dataset Structure：

- **ISIC2018_dataset**
     - test
       	- image
       	- label
     - train
               - image
               - label
     - validation
               - image
               - label



### Code:

- **connect to google drive** (I'm running on Colab)

```
# connect to google drive    use path+ directory
import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/COMP3710/project/ISIC2018_dataset"

os.chdir(path)
os.listdir(path)
```



- **Data preprocessing**:

```
from keras.preprocessing.image import ImageDataGenerator
def trainGenerator(batch_size, train_path, image_folder, mask_folder, data_gen_args, target_size = (128,128),
                   seed = 2):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,#训练数据文件夹路径
        classes = [image_folder],#类别文件夹,对哪一个类进行增强
        color_mode = "rgb",
        class_mode = None,#不返回标签
        target_size = target_size,#读取文件顺便resize的尺寸
        batch_size = batch_size,#每次产生的（进行转换的）图片张数
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        color_mode = "grayscale",
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)#组合成一个生成器

    for (img,mask) in train_generator:   
        '''
        img:(n,128,128,3)   0-255
        mask:(n,128,128,1)
        '''
        img = img/255.0
        mask = mask/255.0
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        yield (img,mask)
    
```

Because there are always a huge number images in dataset, if just read them into the memory at once time, the memory tend to be overflow, so here using [ImageDataGenerator](https://keras.io/api/preprocessing/image/#imagedatagenerator-class), which takes the parameter batch_size as the number of images that the algorithm read from the drive each time.  



- **model**:

  ![Alt text](https://s1.ax1x.com/2020/11/08/BID77j.md.png)
  
  ```
  # Build U-Net model
  from keras import backend as K
  
  def dice_coef(y_true, y_pred):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      intersection = K.sum(y_true_f * y_pred_f)
      return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
  
  
  # Build U-Net model
  from keras.models import *
  from keras.layers import *
  from keras.optimizers import *
  
  
  
  inputs = Input((128,128,3))
  
  
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
  c1 = Dropout(0.1) (c1)
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
  p1 = MaxPooling2D((2, 2)) (c1)
  
  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
  c2 = Dropout(0.1) (c2)
  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
  p2 = MaxPooling2D((2, 2)) (c2)
  
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
  c3 = Dropout(0.2) (c3)
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
  p3 = MaxPooling2D((2, 2)) (c3)
  
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
  c4 = Dropout(0.2) (c4)
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
  p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
  
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
  c5 = Dropout(0.3) (c5)
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
  
  u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
  c6 = Dropout(0.2) (c6)
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
  
  u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
  c7 = Dropout(0.2) (c7)
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
  
  u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
  c8 = Dropout(0.1) (c8)
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
  
  u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
  c9 = Dropout(0.1) (c9)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
  
  outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
  
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=Adam(lr = 3e-4), loss='binary_crossentropy', metrics=[dice_coef])
  
  ```
  
  

Remind that for this task is not includes multiple classes, so in the last layer of Unet,  the activation function should be sigmoid rather than 'softmax' in multiple segmentation task. (In multiple segmentation task, should doing one-hot coding for label, which means extend a channel with the number of class)



- **training**

```
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

print(type(data_gen_args))

X_train = trainGenerator(20,path+'//train','image','label',data_gen_args)
validation_data = trainGenerator(10,path+'//validation','image','label',data_gen_args)

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(X_train,steps_per_epoch=80,epochs=25,callbacks=[model_checkpoint],
                    validation_data = validation_data,
                validation_steps = 60    )
#steps_per_epoch指的是每个epoch有多少个batch_size，也就是训练集总样本数/batch_size
# validation steps = validation data /batch_size
```



We can see from this chart, the dice_loss reached 0.7 at 7 epochs, and finally tend to be stable.

![捕获07](https://s1.ax1x.com/2020/11/08/BIDqNn.md.png)

![捕获08](https://s1.ax1x.com/2020/11/08/BIDbAs.md.png)

![捕获09](https://s1.ax1x.com/2020/11/08/BIDTBQ.md.png)



- **Prediction**

Using ImageDataGenerator to generate test data, and use `model.predict` to get the segmentation result.

Here is a set of pictures to compare the original image and segmentation result.

[![BIrpB4.png](https://s1.ax1x.com/2020/11/08/BIrpB4.png)](https://imgchr.com/i/BIrpB4)

[![BIrSuF.png](https://s1.ax1x.com/2020/11/08/BIrSuF.png)](https://imgchr.com/i/BIrSuF)



