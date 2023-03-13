"""Data loading and preprocessing."""

from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


def get_augmentations(is_train: bool, config) -> keras.Sequential:
    """Get data augmentation layer.

    Args:
        is_train: Whether the augmentation layer is for training or not.
        config: The config dictionary.

    Returns:
        A `keras.Sequential` object containing the data augmentation layer.
    """
    if is_train:
        # Build the `train` augmentation pipeline.
        augmentation_layer = keras.Sequential(
            [
                layers.Rescaling(1 / 255.0, dtype="float32"),
                layers.Resizing(
                    config["input_shape"][0] + 20,
                    config["input_shape"][0] + 20,
                    dtype="float32",
                ),
                layers.RandomCrop(
                    config["image_size"],
                    config["image_size"],
                    dtype="float32",
                ),
                layers.RandomFlip("horizontal", dtype="float32"),
            ],
            name="train_data_augmentation",
        )

    else:
        # Build the `val` and `test` data pipeline.
        augmentation_layer = keras.Sequential(
            [
                layers.Rescaling(1 / 255.0, dtype="float32"),
                layers.Resizing(
                    config["image_size"],
                    config["image_size"],
                    dtype="float32",
                ),
            ],
            name="test_data_augmentation",
        )

    return augmentation_layer


def get_dataset(name: str, is_train: bool):
    """Get a dataset from tensorflow datasets.

    Args:
        name: The name of the dataset.

    Returns:
        A tuple of `tf.data.Dataset` objects."""
    # Load the dataset.
    if name == "cifar10":
        train_ds, val_ds, test_ds = tfds.load(
            name=name,
            split=["train", "test[0:80%]", "test[80%:]"],
        )
    else:
        train_ds, val_ds, test_ds = tfds.load(
            name=name,
            split=["train", "validation[0:80%]", "validation[80%:]"],
        )

    if is_train:
        return (train_ds, val_ds)
    return test_ds


class MapFunction:
    """A wrapper class for data augmentation layer.

    Args:
        augmentation_layer: The data augmentation layer.
    """

    def __init__(self, augmentation_layer: keras.Sequential):
        self.augmentation_layer = augmentation_layer

    def __call__(
        self,
        element,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image = element["image"]
        label = element["label"]
        return self.augmentation_layer(image), label
