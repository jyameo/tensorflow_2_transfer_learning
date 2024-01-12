import tensorflow as tf
from tensorflow.keras import layers, models
from config import (HEIGHT, WIDTH, NUM_CHANNELS, NCLASSES)
from pipeline import input_pipeline

# Get base model from TF HUB
base_model = tf.keras.applications.vgg19.VGG19(
    input_shape=(HEIGHT, WIDTH, NUM_CHANNELS),
    include_top=False,
    weights='imagenet')

base_model.trainable = False

# Extend Base Model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(NCLASSES, activation='sigmoid')(x)

# Define Full Custom Model
model = models.Model(inputs=base_model.input, outputs=x)

# Print Model Architecture
print(model.summary())

# TODO
filepaths = None
labels = None
ds = input_pipeline(filepaths, labels)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds, epochs=3, steps_per_epoch=10)
