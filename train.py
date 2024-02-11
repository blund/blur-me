import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras

import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join



data_dir = "detected-faces-test"

image_height = 64
image_width  = 64
batch_size   = 512

(train_ds, val_ds) = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    validation_split=0.2,
    subset="both",
    seed=123,
    color_mode='grayscale',
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(512, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
  
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Adjust num_classes based on your dataset
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=25
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')

model.save(f'models/bs-{now}.keras')

plt.show()
