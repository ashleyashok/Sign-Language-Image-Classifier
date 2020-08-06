# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:40:17 2020

@author: Ashley
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
    df = pd.read_csv(filename)
	label = df.iloc[:,0].values
	images = df.iloc[:, 1:].values




	return images, labels

print(sys.executable)

path_sign_mnist_train = f"{getcwd()}/../Data/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../Data/sign_mnist_test.csv"


training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

training_images = np.expand_dims(training_images, axis = 3)
testing_images = np.expand_dims(testing_images, axis = 3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(
    rescale = 1./255)

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')]
    )

# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_generator = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=64
)

validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=64
)
# Train the Model
history = model.fit_generator(train_generator,
    epochs=20,
    validation_data=validation_generator)

model.evaluate(testing_images, testing_labels, verbose=0)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc =  history.history['val_accuracy']
loss =  history.history['loss']
val_loss =  history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()