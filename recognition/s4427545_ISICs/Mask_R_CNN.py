import tensorflow as tf
import mrcnn
import mrcnn.config
from mrcnn.model import MaskRCNN as TF2_MaskRCNN
from mrcnn.visualize import display_instances
from skimage import io
import os
import imgaug

# TODO: Python Doc

class BaseConfig(mrcnn.config.Config):
    # static vars for config
    NAME = "ISIC"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = 2
    # Reduce by factor of 10 since our images are relatively simple compared to original 81 class data
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100
    TRAIN_ROIS_PER_IMAGE = 20
    # Final detection box count, keep it low as images are simple
    DETECTION_MAX_INSTANCES = 3

class MaskRCNN():
    CLASS_NAMES = ['BG', 'Lesion']

    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
        self.__build_model()

    def __build_model(self):
        self.model = TF2_MaskRCNN(mode="inference", 
                                    config=BaseConfig(),
                                    model_dir=os.getcwd())
        # Using COCO dataset trained weights, but must exclude some layers as we only have 2 classes
        # instead of 81.
        self.model.load_weights(filepath="./mrcnn/mask_rcnn_coco.h5", 
                        by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # load the input image, convert it from BGR to RGB channel

    def train(self):
        self.print_info()
        train_images, valid_images, train_masks, valid_masks = self.get_and_split_data()
        # Basic 50% left and right flip augmentation
        augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5)])
        self.model.train(train_images, valid_images, BaseConfig.LEARNING_RATE, 10, 'all', augmentation=augmentation)
        
    def get_and_split_data(self):
        train_images = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1-2_Training_Input_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(512, 512),
            shuffle=False, # shuffle false so masks match the images... need a better workaround
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='training') # just do training and valid. for now, work out how to do test later

        valid_images = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1-2_Training_Input_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(512, 512),
            shuffle=False,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation')

        train_masks = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1_Training_GroundTruth_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(512, 512),
            shuffle=False,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='training')

        valid_masks = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1_Training_GroundTruth_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(512, 512),
            shuffle=False,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation')
        # Allows files to be fetched asynchronously
        AUTOTUNE = tf.data.AUTOTUNE
        train_images = train_images.prefetch(buffer_size=AUTOTUNE)
        valid_images = valid_images.prefetch(buffer_size=AUTOTUNE)
        train_masks = train_masks.prefetch(buffer_size=AUTOTUNE)
        valid_masks = valid_masks.prefetch(buffer_size=AUTOTUNE)
        return train_images, valid_images, train_masks, valid_masks
    
    def display_sample(self):
        image = io.imread("sample_image.jpg")
        r = self.model.detect([image], verbose=1)
        r = r[0]
        display_instances(image=image, 
                        boxes=r['rois'], 
                        masks=r['masks'], 
                        class_ids=r['class_ids'], 
                        class_names=self.CLASS_NAMES, 
                        scores=r['scores'])

    def print_info(self):
        print(f"TF Version: {tf.__version__}")
        print(f'GPU: {tf.config.list_physical_devices("GPU")}')
        print(f'Dataset directory: {self.dir}')