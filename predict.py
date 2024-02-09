import os
from os import listdir
from os.path import join

import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
from tensorflow.keras import datasets, layers, models
import keras

import random

path = "/home/blund/skole/23h/bs/detected-faces/"
ids = listdir(path)
id = random.choice(ids)
id_path = join(path, id)
images = listdir(id_path)
image = random.choice(images)
image_path = join(id_path, image)

image = tf.keras.utils.load_img(
    image_path,
    color_mode='grayscale',
    target_size=None,
    interpolation='nearest',
    keep_aspect_ratio=False
)

input_arr = keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

dirs = listdir("models")
dirs.sort()
model_path = f"models/{dirs[-1]}"
model = keras.models.load_model(model_path)

result = model.predict(input_arr)[0]
result = result.tolist()

max_value = max(result)
max_index = result.index(max_value)

print(max_index, max_value)
print(image_path)

print("real id: {id}")
for n in range(9):
    result[max_index] = 0
    max_value = max(result)
    max_index = result.index(max_value)
    print(max_index, max_value)
