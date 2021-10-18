import tensorflow as tf
import mrcnn
import mrcnn.config
from mrcnn.model import MaskRCNN as TF2_MaskRCNN
from mrcnn.visualize import display_instances
from skimage import io
from isics_data_loader import *
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

    def __init__(self, dir, batch_size, valid_split=0.2):
        self.dir = dir
        self.batch_size = batch_size
        self.get_and_split_data(valid_split)
        self.build_model()

    def build_model(self):
        self.model = TF2_MaskRCNN(mode="training", 
                                    config=BaseConfig(),
                                    model_dir=os.getcwd())
        # Using COCO dataset trained weights, but must exclude some layers as we only have 2 classes
        # instead of 81.
        self.model.load_weights(filepath="./mrcnn/mask_rcnn_coco.h5", 
                        by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    def get_and_split_data(self, valid_split):
        training_image_id_map, training_mask_id_map, validation_image_id_map, validation_mask_id_map =\
            training_validation_ids(self.dir, valid_split)
        self.training_data = ISICsDataLoader(training_image_id_map, training_mask_id_map)
        self.validation_data = ISICsDataLoader(validation_image_id_map, validation_mask_id_map)

    def train(self):
        self.print_info()
        # Basic 50% left and right flip augmentation
        augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5)])
        self.model.train(self.training_data, self.validation_data, BaseConfig.LEARNING_RATE, 10, 'all', augmentation=augmentation)
    
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