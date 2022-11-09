import argparse
import os
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt

MAX_COUNT = 11328

def make_path(size, limit, shard, max_shards):
    return f'oasis{size}_{limit}-{shard+1:03}-of-{max_shards:03}.tfrecord'

def thread_fn(args):
    path, images = args
    with tf.io.TFRecordWriter(path) as writer:
        for filename in images:
            img = Image.open(filename)
            img = img.resize((size, size), Image.ANTIALIAS)
            img = np.asarray(img)
            img = img.reshape(size, size, 1)
            img = img.astype(np.float32) / 127.5 - 1
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
            }))
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', help='image size in pixels', type=int)
    parser.add_argument('-n', '--count', help='output n images (shuffled beforehand)', type=int)
    parser.add_argument('-s', '--shards', type=int)
    parser.add_argument('-o', '--output', help='output directory')
    args = parser.parse_args()

    size = args.size
    n_images = args.count if args.count else MAX_COUNT
    n_shards = args.shards if args.shards else 1

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    images_dir = os.path.relpath('data')
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]

    # shuffle the images
    images = np.array(images)
    np.random.RandomState(42).shuffle(images)

    step = n_images // n_shards

    thread_args = []
    for shard in range(n_shards):
        path = make_path(size, n_images, shard, n_shards)
        if args.output:
            path = os.path.join(args.output, path)

        start = shard * step
        end = min((start + step), MAX_COUNT)
        images_shard = images[start:end]
        thread_args.append((path, images_shard))

    with ThreadPoolExecutor() as executor:
        executor.map(thread_fn, thread_args)