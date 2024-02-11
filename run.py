import os
from os import listdir, path

import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models

data_dir = "detected-faces-test"

img_dim    = 64
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='grayscale',
    image_size=(img_dim, img_dim),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_dim, img_dim),
    batch_size=batch_size)

dirs = listdir("models")
dirs.sort()
model_path = f"models/{dirs[-1]}"
model = keras.models.load_model(model_path)
test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(test_loss, test_acc)
