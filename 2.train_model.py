# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:56:49 2023

@author: deben
"""


from tensorflow.keras.models import save_model


import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout,BatchNormalization




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Path to the directory containing the training data
train_path = './Dataset_Creation/train/'

# Get the list of directories inside the training directory
train_dirs = os.listdir(train_path)


# Initialize an empty list to store the images and labels
images = []
labels = []
mapp=[]

# Loop through each directory and read the images
for i, train_dir in enumerate(train_dirs):
    for img_name in os.listdir(train_path + train_dir):
        img = cv2.imread(train_path + train_dir + '/' + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(i)
    mapp.append(train_dir)
# Convert the list of images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the pixel values to be between 0 and 1
images = images / 255.0


# Define the CNN architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_dirs), activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(images, labels, epochs=15, batch_size=64, validation_split=0.2)


# Save this data into file system

# Save this data into file system
np.save('y_train.npy',labels)
np.save('train_labels.npy',mapp)
print("Data Successfully saved")
np.save('x_train.npy',images)
print("Data Successfully saved")


# Save the model as .h5 file
save_model(model, 'trained_model.h5')