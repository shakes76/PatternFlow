from typing import Optional
from pathlib import Path
from utils import Config, set_logger, preview_images
from tensorflow.data.experimental import AUTOTUNE
from model import GAN
import argparse
import logging
import os
import sys
import tensorflow as tf

def parse_tfrecord(record):
    '''
    Parse a TFRecord into an image
    '''
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)
    })

    shape = features['shape']
    data = tf.io.decode_raw(features['data'], tf.float32)
    img = tf.reshape(data, shape)
    return img

if __name__ == '__main__':
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="path to config file", type=str)
    args = parser.parse_args()

    # Load configuration file
    if args.config:
        config.load_json(args.config)

    output_dir = config.output_dir
    input_dir = config.input_dir
    output_dir.mkdir(exist_ok=False)

    # Setup logging
    set_logger(output_dir / 'train.log')

    strategy = tf.distribute.MirroredStrategy()

    config.global_batch_size = config.batch_size * strategy.num_replicas_in_sync

    # Create input pipeline 
    logging.info('Loading input files...')
    files = tf.io.matching_files(f'{input_dir}/*.tfrecord')
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
	.cache()
        .shuffle(config.buffer_size)
        .batch(config.global_batch_size)
        .prefetch(AUTOTUNE)
    )

    dataset = strategy.experimental_distribute_dataset(dataset)

    # Begin training
    logging.info('Training...')
    with strategy.scope():
        gan = GAN(config)

        gan.generator.summary()
        gan.discriminator.summary()
        gan.train(dataset, config.epochs)

        preview = gan.generator.predict(config.fixed_seed)
        image = preview_images(config, preview, config.epochs)
        image.save(config.output_dir / "final.png")

        gan.generator.save(config.output_dir / 'models' / 'generator')
        gan.discriminator.save(config.output_dir / 'models' / 'discriminator')
