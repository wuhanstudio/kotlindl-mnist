#!/usr/bin/env python
# coding: utf-8

import time

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential(
[
    Convolution2D(32, 3, activation="relu", input_shape=(32, 32, 3), 
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Convolution2D(64, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    Convolution2D(128, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    Convolution2D(128, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Convolution2D(256, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    Convolution2D(256, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Convolution2D(256, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    Convolution2D(256, 3, activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    layers.Flatten(),

    layers.Dense(512, activation='relu', 
        kernel_initializer="he_normal",
        bias_initializer="he_normal"),
    layers.Dense(512, activation='relu', 
        kernel_initializer="he_normal",
        bias_initializer="he_normal"),
    layers.Dense(10, activation='linear', 
        kernel_initializer="he_normal",
        bias_initializer="he_normal")
])

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

start = time.time()
history = model.fit(train_images, train_labels, epochs=3, batch_size=128,
                    validation_data=(test_images, test_labels))
print("Training time: ", time.time() - start)

test_loss, test_acc = model.evaluate(test_images,  test_labels, batch_size=1000)
print("Accuracy: ", test_acc)
