"""
Alien Attack War Events Image Classification

Summary:
This script trains a convolutional neural network (CNN) to classify images depicting various war events 
caused by alien attacks on the planet. The dataset consists of images categorized into five classes:
1. Alien Invasion
2. Battle with Alien Spaceships
3. Destruction of Cities by Aliens
4. Defense Force Engaging Aliens
5. Aftermath of Alien Attack

The model architecture used is a CNN, known for its effectiveness in image classification tasks. 
The dataset is assumed to be structured in directories with each class having its own subdirectory.

Author: ARsenic100
Date: June 20, 2024
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential

combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

img_height = 180
img_width = 180
batch_size = 32

# Using tf.keras.utils.image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    "training",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "training",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Data preprocessing
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Cache and prefetch for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define the model
num_classes = 5  # Number of classes
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Added dropout layer
    layers.Dense(num_classes, activation='softmax')
])

# Learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
epochs = 75
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save the model with information about the number of epochs
model.save(f"modelpr.keras")


