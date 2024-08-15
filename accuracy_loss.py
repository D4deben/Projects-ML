# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:14:11 2023

@author: deben
"""

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.keras.models import load_model



class LossAccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.test_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.test_acc.append(logs.get('val_accuracy'))
        

# Load the pre-trained Keras model
model = load_model('trained_model.h5')


# Create an instance of the LossAccuracyHistory callback
history = LossAccuracyHistory()

#load trained and test data
x_train = np.memmap('x_train.npy', dtype='float64', mode='r', shape=(199360512,))
y_train= np.load('y_train.npy')
x_test= np.load('x_test.npy')
y_test= np.load('y_test.npy')

# Assuming you have already compiled and trained your model using fit()
model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[history], epochs=20, batch_size=64, validation_split=0.2)

# Access the recorded loss and accuracy values from the history object
train_loss = history.train_loss
test_loss = history.test_loss
train_acc = history.train_acc
test_acc = history.test_acc

# Plot the training and test loss values
epochs = range(1, len(train_loss) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, test_loss, 'ro-', label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and test accuracy values
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, test_acc, 'ro-', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
