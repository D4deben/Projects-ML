# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:33:26 2023

@author: deben
"""
from tensorflow.keras.models import load_model
import cv2
import numpy as np

import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Load the trained CNN model from .h5 file
model = load_model('trained_model.h5')

# Load an image for face recognition
##image = cv2.imread('test_image.jpg')

# Preprocess the image
# Perform any necessary preprocessing steps such as resizing, normalizing, etc.

# Perform face recognition
# Assuming the model expects input of shape (batch_size, height, width, channels)

# Process the output
# Depending on your model's architecture and task, process the output accordingly
# For example, if the output represents class probabilities, you can use argmax to get the predicted class

# Perform further actions based on the output
# Use the output to make decisions or take actions based on your application's requirements




# Path to the directory containing the training data
test_path = './test/'

# Get the list of directories inside the training directory
test_dirs = os.listdir(test_path)

# Initialize an empty list to store the images and labels
test_images = []
test_labels = []


# Loop through each directory and read the images
for i, test_dir in enumerate(test_dirs):
    for img_name in os.listdir(test_path + test_dir):
        img = cv2.imread(test_path + test_dir + '/' + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x,y,w,h) in faces:
            # Extract the face from the image
            face = img[y:y+h, x:x+w]

            # Resize the face to match the input size of the CNN model
            face = cv2.resize(face, (600,600))

            # Normalize the pixel values to be between 0 and 1
            face = face / 255.0

            # Add a batch dimension to the face array
            face = np.expand_dims(face, axis=0)
            #cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Display the output image
        cv2.imshow('Output Image', img)
        cv2.waitKey(200)
        cv2.destroyAllWindows()

        img = cv2.resize(img, (128, 128))
        test_images.append(img)
        test_labels.append(i)
    

# Convert the list of images and labels to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize the pixel values to be between 0 and 1
test_images = test_images / 255.0 

# Save this data into file system
np.save('y_test.npy',test_labels)
print("Data Successfully saved")
np.save('x_test.npy',test_images)
print("Data Successfully saved")


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
