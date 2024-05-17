
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf


data = mnist.load_data()

(x_train, y_train), (x_test, y_test) = data
x_train = x_train / 255.0
x_test = x_test / 255.0

## Extraction of classes

x_train_01 = np.where((y_train == 0) | (y_train == 1))
x_test_01 = np.where((y_test == 0) | (y_test == 1))


X_train, X_test = x_train[x_train_01], x_test[x_test_01]
Y_train, Y_test = y_train[x_train_01], y_test[x_test_01]


def downsample(image, newshape):
    image = image[..., np.newaxis]
    image = tf.image.resize(images=image, size=newshape)
    image = tf.reshape(image, (image.shape[0]*image.shape[1], 1))
    return image


new_x_train = []
for x in X_train:
    x = downsample(x, (16, 16))
    new_x_train.append(x)

new_x_test = []
for x in X_test:
    x = downsample(x, (16, 16))
    new_x_test.append(x)

new_x_train = np.array(new_x_train)
new_x_test = np.array(new_x_test)

new_x_train = np.squeeze(new_x_train)
new_x_test = np.squeeze(new_x_test)

print(new_x_train.shape)