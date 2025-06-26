import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split



with_mask = os.listdir('/Users/nithunsundarrajan/Downloads/data/with_mask')
without_mask = os.listdir('/Users/nithunsundarrajan/Downloads/data/without_mask')

print(with_mask[:5])
print(without_mask[:5])

print(len(with_mask))
print(len(without_mask))

#create the labels
with_mask_labels = [1 for i in range(len(with_mask))]
without_mask_labels = [0 for i in range(len(without_mask))]

labels = with_mask_labels + without_mask_labels
print(len(labels))
print(labels[0:5])
print(labels[-5:])

#image processing 

# ...existing code...

with_mask_path = '/Users/nithunsundarrajan/Downloads/data/with_mask'
without_mask_path = '/Users/nithunsundarrajan/Downloads/data/without_mask'

data = []
labels = []

# Process with_mask images
for img_file in with_mask:
    image = Image.open(os.path.join(with_mask_path, img_file))
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)
    labels.append(1)  # label for with_mask

# Process without_mask images
for img_file in without_mask:
    image = Image.open(os.path.join(without_mask_path, img_file))
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)
    labels.append(0)  # label for without_mask

data = np.array(data)
labels = np.array(labels)

print(data.shape)
print(labels.shape)


X = data
Y = labels
#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

#Building a CNN model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
     

# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
     

# training the neural network
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=5)

loss, accuracy = model.evaluate(X_test, Y_test)
print('Test Accuracy =', accuracy)

h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()


