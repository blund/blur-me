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

from db import *

# Load your trained model
dirs = listdir("models")
dirs.sort()
model_path = f"models/{dirs[-1]}"
model = keras.models.load_model(model_path)

# Access the layers of the model
layers = model.layers

# Remove the last layer
new_layers = layers[:-1]  # Removes the last layer

# Cut of last layer
new_model = tf.keras.models.Sequential(new_layers)

# Optionally, you can freeze the layers of the new model to prevent them from being trained
for layer in new_model.layers:
    layer.trainable = False

# Compile the new model if needed
new_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


id = 30
image = 1

def get_image(id, number_index):

    number_index = number_index
    
    path = "/home/blund/skole/23h/bs/detected-faces-test/"
    ids = listdir(path)
    id_path = join(path, id)
    images = listdir(id_path)
    image = images[number_index] # random.choice(images)
    image_path = join(id_path, image)
    image_loaded = tf.keras.utils.load_img(
        image_path,
        color_mode='grayscale',
        target_size=None,
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    input_arr = keras.utils.img_to_array(image_loaded)
    return np.array([input_arr])  # Convert single image to a batch.


# db = create()
# for i in range(25,33):
#     img = get_image(f"0000{i}", 0)
#     attributes  = new_model.predict(img)[0]
#     db = add_entry(db, attributes)

    
# save(db, "db.npy")

db = load("db.npy")
print(db)

img = get_image("000025", 2)
attributes  = new_model.predict(img)[0]

(id, distance) = query(db, attributes)

print(id, distance)

# print(1-cosine_similarity(result_a, result_b))
# print(1-cosine_similarity(result_a, result_c))
# print(1-cosine_similarity(result_b, result_c))

