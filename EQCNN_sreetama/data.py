# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import h5py
import pickle
import numpy as np
import tensorflow as tf
import json
#from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.decomposition import PCA

import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms

import os
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

def data_load_and_process(dataset, classes=[0, 1], feature_reduction="resize1024", binary=True):
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data
        
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data
        
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize the data
        
            
    elif dataset == "bloodmnist":
        data_flag = "bloodmnist"
        download = True
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        import torch.utils.data as data
        # preprocessing
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        train_dataset = DataClass(split='train', transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size = len(train_dataset))
        test_loader = data.DataLoader(dataset=test_dataset, batch_size = len(test_dataset))
        for x, y in train_loader:
            x1_train=x.numpy()
            y1_train=y.numpy()
            x_train = x1_train.squeeze()
            y_train = y1_train.squeeze()
            x_train = np.transpose(x_train, (0, 2, 3, 1))
        for x, y in test_loader:
            x1_test=x.numpy()
            y1_test=y.numpy()
            x_test = x1_test.squeeze()
            y_test = y1_test.squeeze()
            x_test = np.transpose(x_test, (0, 2, 3, 1))
            
    
    if dataset=="mnist" or dataset=="fashion_mnist" or dataset=="cifar10" or dataset=="bloodmnist":
        x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
        x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

        X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
        Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]
        

    if binary == False:
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
    elif binary == True:
        Y_train = [1 if y == classes[0] else -1 for y in Y_train]  # kept the labels as [0, 1] intact
        Y_test = [1 if y == classes[0] else -1 for y in Y_test]
        
    
    
    if feature_reduction == 'resize256':
        if dataset=="cifar10":
            X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
            X_train = tf.image.rgb_to_grayscale(X_train[:])
            X_test = tf.image.rgb_to_grayscale(X_test[:])
            X_train = tf.image.resize(X_train[:], (32, 32)).numpy()
            X_test = tf.image.resize(X_test[:], (32, 32)).numpy()
            #pad1 = np.full((np.shape(X_train)[0], 32, 32, 1), 0)
            #pad2 = np.full((np.array(np.shape(X_test))[0], 32, 32, 1), 0)
            #X_train = np.append(X_train, pad1, axis=3)
            #X_test = np.append(X_test, pad2, axis=3)
            X1_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 1024, 1, 1))
            X1_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 1024, 1, 1))
            X1_train, X1_test = tf.squeeze(X1_train).numpy(), tf.squeeze(X1_test).numpy()
                
        
        elif dataset == "mnist" or dataset == "fashion_mnist":
            X_train = tf.image.resize(X_train[:], (4, 4)).numpy()
            X_test = tf.image.resize(X_test[:], (4, 4)).numpy()
            #X1_train = tf.reshape(X_train[:], (np.shape(X_train)[0], 256, 1))
            #X1_test = tf.reshape(X_test[:], (np.shape(X_test)[0], 256, 1))
            X1_train, X1_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
        
    return X1_train, X1_test, Y_train, Y_test
        
 
     
    
