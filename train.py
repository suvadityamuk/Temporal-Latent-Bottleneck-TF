# USAGE:
# python train.py --dataset <dataset_name> --mixed-precision <True/False>

import argparse

import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW

from configs import config_cifar, config_imagenette
from tlb import (
    CustomRecurrentCell,
    MapFunction,
    ModelTrainer,
    PatchEmbed,
    get_augmentations,
    get_dataset,
)

AUTO = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--mixed-precision", type=bool, default=False)
    args = parser.parse_args()
    return args


def run(args: argparse.Namespace):
    # Load the config.
    if args.dataset == "cifar10":
        config = config_cifar.config
    else:
        config = config_imagenette.config

    # Set the mixed precision policy.
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    # Load the dataset and get the augmentation layer.
    print("Loading the dataset...")
    (train_ds, val_ds) = get_dataset(name=config["dataset_name"], is_train=True)

    # Build the augmentation layer.
    print("Building the augmentation layer...")
    train_augmentation = get_augmentations(is_train=True, config=config)
    test_augmentation = get_augmentations(is_train=False, config=config)

    # Build the data pipeline.
    print("Building the data pipeline...")
    train_map_function = MapFunction(augmentation_layer=train_augmentation)
    test_map_function = MapFunction(augmentation_layer=test_augmentation)

    # Build the `train` and `val` data pipeline.
    train_ds = (
        train_ds.map(train_map_function, num_parallel_calls=AUTO)
        .shuffle(config["buffer_size"])
        .batch(config["batch_size"], num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )
    val_ds = (
        val_ds.map(test_map_function, num_parallel_calls=AUTO)
        .batch(config["batch_size"], num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    # Build the model.
    print("Building the model...")
    patch_layer = PatchEmbed(
        image_size=(config["image_size"], config["image_size"]),
        patch_size=(config["patch_size"], config["patch_size"]),
        embed_dim=config["embed_dim"],
        chunk_size=config["chunk_size"],
    )
    custom_rnn_cell = CustomRecurrentCell(
        chunk_size=config["chunk_size"],
        r=config["r"],
        num_layers=config["num_layers"],
        ffn_dims=config["embed_dim"],
        ffn_dropout=config["ffn_drop"],
        num_heads=config["num_heads"],
        key_dim=config["embed_dim"],
        attn_dropout=config["attn_drop"],
    )
    model = ModelTrainer(patch_layer=patch_layer, custom_cell=custom_rnn_cell)

    # Compile the model.
    print("Compiling the model...")
    optimizer = AdamW(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model.
    print("Training the model...")
    history = model.fit(
        train_ds,
        epochs=config["epochs"],
        validation_data=val_ds,
    )

    return history


if __name__ == "__main__":
    args = parse_args()
    history = run(args)
