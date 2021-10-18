#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential(
[
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256),
    layers.Dense(128),
    layers.Dense(10, activation='softmax')
])


model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
