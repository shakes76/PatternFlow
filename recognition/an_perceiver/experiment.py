"""Perceiver classification experiment on AOI AKOA knee MRI slices.

This script is intended to be run from command line and accepts parameters for
configuring the the perceiver model configuration, dataset location,
*some* pre-processing options and training options are also available.

Run `python experiment.py --help` for more information.

The pre-defined defaults are *mostly* consistent with the configuration for
imagenet classifier described in the Perceiver paper https://arxiv.org/abs/2103.03206

@author Anthony North
"""

import argparse
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import callbacks, losses, metrics
from tensorflow_addons import optimizers

import aoi_akoa  # register dataset
from perceiver import Perceiver
from preprocessing import preprocess


def run_experiment(opts):
    """Run perceiver classification experiment on aoi_akoa dataset."""

    splits, info = tfds.load(
        "aoi_akoa",
        split=["train", "validation", "test"],
        data_dir=opts.data_dir,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    num_classes = info.features["label"].num_classes
    train, validation, test = preprocess(
        *splits,
        num_classes=num_classes,
        image_dims=opts.image_dims,
        hflip_concat=opts.hflip_concat,
        train_batch_size=opts.train_batch_size,
        eval_batch_size=opts.eval_batch_size,
    )

    perceiver = Perceiver(
        num_blocks=opts.num_blocks,
        num_self_attends_per_block=opts.num_self_attends_per_block,
        num_cross_heads=opts.num_cross_heads,
        num_self_attend_heads=opts.num_self_attend_heads,
        latent_dim=opts.latent_dim,
        latent_channels=opts.latent_channels,
        num_freq_bands=opts.num_freq_bands,
        num_classes=num_classes,
    )

    perceiver.compile(
        optimizer=optimizers.LAMB(
            learning_rate=opts.learning_rate,
            weight_decay_rate=opts.weight_decay_rate,
        ),
        loss=losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=opts.label_smoothing
        ),
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
            # paper uses this metric, redundant for 2-class however
            metrics.TopKCategoricalAccuracy(1, name="top1_accuracy"),
        ],
    )

    csv_logger = callbacks.CSVLogger(filename=os.path.join(opts.out_dir, "history.csv"))
    model_checkpointer = callbacks.ModelCheckpoint(
        filepath=os.path.join(opts.out_dir, "checkpoint"), save_weights_only=True
    )

    history = perceiver.fit(
        train,
        batch_size=opts.train_batch_size,
        validation_data=validation,
        validation_batch_size=opts.train_batch_size,
        epochs=opts.epochs,
        callbacks=[model_checkpointer, csv_logger],
    )

    perceiver.save_weights(os.path.join(opts.out_dir, "perceiver"))
    eval_result = perceiver.evaluate(test, return_dict=True)

    with open(os.path.join(opts.out_dir, "eval.txt"), "w") as file:
        print(eval_result, file=file)

    print("\n", "evaluation:", eval_result)


def get_opts():
    """Parses command line options."""

    parser = argparse.ArgumentParser(
        description="Perceiver classification experiment for AOI AKOA knee laterality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # other
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        help="tensorflow log level",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        default=True,
        help="show experiment config",
    )

    # training options
    training = parser.add_argument_group("training options")
    training.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs",
    )
    training.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="batch size for train and validation splits during training",
    )
    training.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="batch size for test during evaluation",
    )
    training.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="label smoothing for categorical loss",
    )
    training.add_argument(
        "--out-dir",
        type=str,
        default="./training",
        help="dir for training outputs",
    )

    # dataset options
    dataset = parser.add_argument_group("dataset options")
    dataset.add_argument(
        "--data-dir",
        type=str,
        default="~/tensorflow_datasets",
        help="location of tensorflow datasets data",
    )
    dataset.add_argument(
        "--image-dims",
        type=int,
        nargs="+",
        help="resize images without distortion",
    )
    dataset.add_argument(
        "--hflip-concat",
        action="store_true",
        default=False,
        help="train and validation: horizontally flip images, reverse labels and concat with input",
    )

    # perceiver options
    perceiver = parser.add_argument_group("perceiver options")
    perceiver.add_argument(
        "--num-blocks",
        type=int,
        default=8,
        help="number of blocks",
    )
    perceiver.add_argument(
        "--num-self-attends-per-block",
        type=int,
        default=6,
        help="number of self attention layers per block",
    )
    perceiver.add_argument(
        "--num-cross-heads",
        type=int,
        default=1,
        help="number of cross attention heads",
    )
    perceiver.add_argument(
        "--num-self-attend-heads",
        type=int,
        default=8,
        help="number of self attention heads",
    )
    perceiver.add_argument(
        "--latent-dim",
        type=int,
        default=512,
        help="latent array dimension",
    )
    perceiver.add_argument(
        "--latent-channels",
        type=int,
        default=1024,
        help="latent array channels",
    )
    perceiver.add_argument(
        "--num-freq-bands",
        type=int,
        default=64,
        help="frequency bands for fourier position encoding",
    )

    # optimiser options
    optimiser = parser.add_argument_group("optimiser options")
    optimiser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-3,
        help="learning rate",
    )
    optimiser.add_argument(
        "--weight-decay-rate",
        type=float,
        default=1e-1,
        help="weight decay rate",
    )

    args = parser.parse_args()
    assert args.image_dims is None or len(args.image_dims) == 2
    if args.image_dims is not None:
        args.image_dims = tuple(args.image_dims)

    return args


def main():
    opts = get_opts()
    tf.get_logger().setLevel(opts.log_level)

    if opts.show_config:
        print("experiment config:")
        for arg in vars(opts):
            print(f"  {arg}:", getattr(opts, arg))

    run_experiment(opts)


if __name__ == "__main__":
    main()
