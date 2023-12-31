import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

def zero_centered(data):
    return ((np.array(data) / 255) - 0.5) * 2

def zero_centered_normalization(train, test, val):
    return zero_centered(train), zero_centered(test), zero_centered(val)

def standard_normalization(train, test, val):
    return np.array(train)/255, np.array(test)/255, np.array(val)/255

def flatten(data):
    return data.reshape(data.shape[0], -1)

def data_flatten(train, test, val):
    return flatten(train), flatten(test), flatten(val)

def norm2(data, img_size):
    #add one channel based on CNN
    return data.reshape(-1, img_size, img_size, 1)

def data_norm_cnn(train, test, val, img_size):
    return norm2(train, img_size), norm2(test, img_size), norm2(val, img_size)

def one_hot_encoding(y_train, y_test, y_val):
    return to_categorical(y_train), to_categorical(y_test), to_categorical(y_val)
