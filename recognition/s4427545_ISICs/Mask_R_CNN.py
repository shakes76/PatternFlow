import tensorflow as tf
import mrcnn
import mrcnn.config
from mrcnn.model import MaskRCNN as TF2_MaskRCNN
from mrcnn.visualize import display_instances
import csv
import os

class BaseConfig(mrcnn.config.Config):
    # static vars for config
    NAME = "ISIC"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = 2

class MaskRCNN():
    CLASS_NAMES = ['BG', 'Lesion']

    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size

    def train(self):
        self.print_info()
        images = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1-2_Training_Input_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(1024, 1024),
            shuffle=True,
            batch_size=self.batch_size)
        masks = tf.keras.preprocessing.image_dataset_from_directory(
            self.dir + 'ISIC2018_Task1_Training_GroundTruth_x2',
            label_mode=None,
            color_mode='rgb',
            seed=42,
            image_size=(1024, 1024),
            shuffle=True,
            batch_size=self.batch_size)
        self.build_model()

    def build_model(self):
        model = TF2_MaskRCNN(mode="inference", 
                                    config=BaseConfig(),
                                    model_dir=os.getcwd())
        # Using COCO dataset trained weights
        model.load_weights(filepath="./mrcnn/mask_rcnn_coco.h5", 
                        by_name=True)
        # load the input image, convert it from BGR to RGB channel
        image = csv.imread("sample_image.jpg")
        image = csv.cvtColor(image, csv.COLOR_BGR2RGB)
        r = model.detect([image], verbose=1)
        r = r[0]
        # visualise result
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

    def generate_bounding_boxes(self, images):
        coords = []
        for image in images:
            pass