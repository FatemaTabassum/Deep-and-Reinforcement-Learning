#!/usr/bin/env python
# coding: utf-8

#   #-------------- SAVE DATA ---------------- #

# In[6]:



from numpy import mean
from numpy import std
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import KFold
from keras.optimizers import SGD
import tensorflow as tf
import keras
from keras.models import Model
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import random


# load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

img_x, img_y = 28, 28

X = np.empty([0, X_train.shape[1], X_train.shape[2], X_train.shape[3]])
print(X.shape)
for i in range(0, X_train.shape[0]):
	if y_train[i] == 6:
		x = np.append(X, [X_train[i]], 0)
		break
print(x.shape)
plt.imshow(x[0, :, :, 0], cmap="gray")
plt.show()

for j in range(0, img_x-12):
    for k in range(0, img_y-12):
        x = np.append(x, [X_train[i]], 0)
        for m in range(0, patch_row):
            for p in range(0, patch_col):
                x[-1][j+m][k+p][0] = 0
        plt.imshow(x[-1, :, :,0], cmap="gray")
        plt.savefig("/Users/liza/Documents/deep 2/homework 3/Images/x_"+str(j)+"_"+str(k)+".png")
        plt.cla()
        
print(x.shape)
np.save("/Users/liza/Documents/deep 2/homework 3/Images/x.npy", x)





# choose random instances and generate advarsarial
ix = random.randint(0, X_train.shape[0])
rand_data = X_train[ix]
rand_data = np.expand_dims(rand_data, axis=0)
x1 = np.zeros((28, 28, 1),  dtype=int)
x1 = np.expand_dims(x1, axis=0)
print(x1.shape)
patch_row, patch_col = 8, 8
img_x, img_y = 28, 28


X = np.empty([0, X_train.shape[1], X_train.shape[2], X_train.shape[3]])
print(X.shape)
for i in range(0, X_train.shape[0]):
    if y_train[i] == 6:
        x = np.append(X, [X_train[i]], 0)
        break
        
for l in range(0, 20):
    rand_pos = random.randint(0, 16)
    x1 = np.append(x1, [rand_data[0]], 0)
    for m in range(0, patch_row):
        for p in range(0, patch_col):
            x1[-1][rand_pos + m][rand_pos + p][0] = rand_data[-1][rand_pos + m][rand_pos + p][0]

for jj in range(0, img_x-12):
    for kk in range(0, img_y-12):
        x = np.append(x, [X_train[i]], 0)
        for mm in range(0, patch_row):
            for pp in range(0, patch_col):
                x[-1][jj+mm][kk+pp][0] = x1[-1][jj+mm][kk+pp][0]
                
        plt.imshow(x[-1, :, :,0], cmap="gray")
        plt.savefig("/Users/liza/Documents/deep 2/homework 3/Images/x_"+"adv_filter_"+str(jj)+"_"+str(kk)+".png")
        plt.cla()

        
np.save("/Users/liza/Documents/deep 2/homework 3/Images/x_adv.npy", x)


# In[ ]:




