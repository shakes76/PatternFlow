from keras.preprocessing.image import ImageDataGenerator


class ImageLoader():

    def __init__(self, dir, mode='RGB'):
        self.dir = dir
        self.mode = mode
        self.gen = ImageDataGenerator(preprocessing_function=self.norm_img)

    def norm_img(self, img):
        img = img.astype('float32')
        img = img/127.5 - 1
        return img

    def load(self, n, res):
        return self.gen.flow_from_directory(batch_size=n, directory=self.dir, target_size=res, color_mode=self.mode, class_mode='binary')
