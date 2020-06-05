#!/usr/bin/env python
# coding: utf-8

# In[47]:


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


# In[48]:


print(tf.__version__)


# In[49]:


def data_loader():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, y_test

def reshape_data(X_train, y_train, X_test, y_test):
    #reshape data to fit model
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    #One hot-encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test

def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def baseline_model():
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, kernel_size=5, activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#              # --------------------- TASK-1-PART-1---------------------#
#     
#         # --------------------------- Visualize Filters -----------------------------------#

# In[59]:




def visualize_filters():
    model = baseline_model()
    conv_layer_cnt = 0;
    plt_cnt = 1
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        print(layer.name)
        # get filter weights
        filters, biases = layer.get_weights()
        print(filters.shape)
        if conv_layer_cnt < 3:
            #print(filters)
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            n_filters, ix = filters.shape[3], 1
            row, col = 0, 0
            if n_filters == 32:
                row, col = 8, 4
            if n_filters == 64:
                row, col = 8, 8
            if n_filters == 128:
                row, col = 8, 6
            
            cnt = 0
            fig, ax = plt.subplots(row, col, figsize = (8,8))
            plt.tight_layout()
            for i in range(row):
                for j in range(col):
                    f = filters[:, :, :, cnt]
                    ax[i][j].set_xticks([])
                    ax[i][j].set_yticks([])
                    ax[i][j].set_title('filter'+ str(cnt))
                    ax[i][j].imshow(f[:, :, 0], cmap='gray')
                    cnt += 1
                    ix += 1
            plt.savefig('conv'+str(plt_cnt)+'.pdf', dpi=300)
            plt_cnt += 1

        # show the figure 
        plt.show()
        conv_layer_cnt += 1
        
visualize_filters()


# In[60]:



                 # --------------------- PROBLEM-1-TASK-2 ----------------#


            #-------------------------- Feature Map -------------------------------#


# In[61]:


# load dataset
X_train, y_train, X_test, y_test = data_loader()
# reshape data
X_train, y_train, X_test, y_test = reshape_data(X_train, y_train, X_test, y_test)
# prepare pixel data
X_train, X_test = prep_pixels(X_train, X_test)
# load the model
model_baseline = baseline_model()
history = model_baseline.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
model_baseline.summary()
predictions = model_baseline.predict(X_test)
print(predictions)
preds_classes = np.argmax(predictions, axis=-1)
print(preds_classes)


# In[63]:


input_shape = (28, 28, 1)
data = np.empty((0,)+input_shape)
print(data.shape)

for i in range(0, preds_classes.shape[0]):
    if(preds_classes[i] == np.argmax(y_test[i]) and preds_classes[i] == 0):
        data = np.append(data, [X_test[i]], 0)
        break
for i in range(0, preds_classes.shape[0]):
    if(preds_classes[i] == np.argmax(y_test[i]) and preds_classes[i] == 8):
        print(' h ', y_test[i], ' h ', preds_classes[i])
        data = np.append(data, [X_test[i]], 0)
        break
for i in range(0, data.shape[0]): 
    c = data[i,:,:,0]
    print(c.shape)
    plt.imshow(data[i,:,:,0], cmap="gray") # data bujhte hobe
    plt.show()


# In[64]:



  
#---------------------- Summary check -------------#
model_baseline.summary()
#--------------------------------------------------#



# In[65]:


feature_map_model = Model(inputs=model_baseline.input, outputs=model_baseline.layers[3].output)
feature_map_model.summary()

second_convolutional_feature_map_model = Model(inputs=model_baseline.input,
                                   outputs=model_baseline.layers[1].output)

first_convolutional_feature_map_model = Model(inputs=model_baseline.input,
                                  outputs=model_baseline.layers[0].output)

first_maxpool_feature_map_model = Model(inputs=model_baseline.input,
                                     outputs=model_baseline.layers[2].output)

first_convolutional_feature_map = first_convolutional_feature_map_model.predict(data)
second_convolutional_feature_map = second_convolutional_feature_map_model.predict(data)
first_maxpool_feature_map = first_maxpool_feature_map_model.predict(data)
feature_map = feature_map_model.predict(data)

print(data.shape)
print(first_convolutional_feature_map.shape)
print(second_convolutional_feature_map.shape)


# In[66]:


for i in range(len(model_baseline.layers)):
	layer = model_baseline.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
    
def find_row_col(n_filters):
    row, col = 0, 0 
    if n_filters==32:
        row, col = 4, 8
    if n_filters==64:
        row, col = 8, 8
    if n_filters==128:
        row, col = 8, 6
    return row, col

no_of_image = data.shape[0]


# In[95]:


print('first_convolutional_feature_map ')           
for i in range(no_of_image):
    no_of_filter = first_convolutional_feature_map.shape[3]
    row, col = find_row_col(no_of_filter)
    ix = 1
    print("first_convolutional_feature_map ", i)
    for _ in range(row):
        for _ in range(col):
            
            # specify subplot and turn of axis
            ax = pyplot.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(first_convolutional_feature_map[i, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    
    pyplot.imshow(first_convolutional_feature_map[1, :, :, 8], cmap='gray')
    pyplot.show()


# In[92]:



print('second_convolutional_feature_map  ')           
for i in range(no_of_image):
    no_of_filter = second_convolutional_feature_map .shape[3]
    row, col = find_row_col(no_of_filter)
    ix = 1
    print("second_convolutional_feature_map  ", i)
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = pyplot.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(second_convolutional_feature_map [i, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    pyplot.imshow(second_convolutional_feature_map[1, :, :, 8], cmap='gray')
    pyplot.show()


    


# In[93]:


print('first_maxpool_feature_map  ')           
for i in range(no_of_image):
    no_of_filter = first_maxpool_feature_map .shape[3]
    row, col = find_row_col(no_of_filter)
    ix = 1
    print("first_maxpool_feature_map  ", i)
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = pyplot.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(first_maxpool_feature_map [i, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    pyplot.imshow(first_maxpool_feature_map[1, :, :, 8], cmap='gray')
    pyplot.show()


# In[94]:


print('feature_map  ')           
for i in range(no_of_image):
    no_of_filter = feature_map .shape[3]
    row, col = find_row_col(no_of_filter)
    ix = 1
    print("feature_map  ", i)
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = pyplot.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_map [i, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    pyplot.imshow(feature_map[1, :, :, 8], cmap='gray')
    pyplot.show()


# In[74]:



                     #--------------------- PROBLEM-1-TASK-3 ----------------#

data = np.empty((0,)+input_shape)

predictions = model_baseline.predict(X_test)
for i in range(0, predictions.shape[0]):
    if(np.argmax(predictions[i]) == np.argmax(y_test[i]) and np.argmax(y_test[i]) == 1):
        data = np.append(data, [X_test[i]], 0)
        break

print(data.shape, " ", X_train.shape)

datagen = ImageDataGenerator()

plt.imshow(data[0, :, :,0], cmap="gray")
plt.xticks(np.arange(0, 28+1, 3.0))
plt.yticks(np.arange(0, 28+1, 3.0))
plt.show()

datagen.fit(X_train)

transform_parameters_left = { 'ty': 3}
x_transform = datagen.apply_transform(data[0], transform_parameters_left)
plt.imshow(x_transform[ :, :,0], cmap="gray")
plt.xticks(np.arange(0, 28+1, 3.0))
plt.yticks(np.arange(0, 28+1, 3.0))
plt.show()

data = np.append(data, [x_transform], 0)
print(data.shape)

transform_parameters_right = { 'ty': -3}
x_transform = datagen.apply_transform(data[0], transform_parameters_right)
plt.imshow(x_transform[ :, :,0], cmap="gray")
plt.xticks(np.arange(0, 28+1, 3.0))
plt.yticks(np.arange(0, 28+1, 3.0))
plt.show()

data = np.append(data, [x_transform], 0)
print(data.shape)

y_pred = model_baseline.predict(data)

print("Prediction for 1: ", np.argmax(y_pred[0])," ", np.max(y_pred[0]))
print("Prediction for left shifted 1: ", np.argmax(y_pred[1])," ", np.max(y_pred[1]))
print("Prediction for right shifted 1: ", np.argmax(y_pred[2])," ", np.max(y_pred[2]))

                #-------------------------- end  -------------------------------#


#                     
#                     #--------------------- PROBLEM-2-TASK-(1-3) ----------------#
# 

# In[85]:


'''Here we empirically test the robustness of the model you have for Problem 1. Imagine that we have a 8x8 black patch and we move it from left to right, from top to down with a stride of 1 to occlude part of the image. Using a digit ‘6’ (which is classified correctly initially) 
as an example, answer the following questions.
(1) Create three maps (i.e., images) in the following way.
For each position of the black patch, store the probability of ‘6’ 
of the partially covered image in map 1, the highest probability (among the 10 classes) 
in map 2, and classified label (‘0’ to ‘9’) in map 3. Display the maps. Make sure that they
are clearly legible by scaling values.'''


x_map = np.load("Images/x.npy")
x_adv = np.load("Images/x_adv.npy")
#print(model.evaluate(x_test, y_test))

y_pred = model_baseline.predict(x_map)
#y_pred = np.argmax(y_pred, axis = -1)
y_pred_adv = model_baseline.predict(x_adv)
#y_pred_adv = np.argmax(y_pred_adv, axis = -1)
#print(y_pred_adv)

correct_list = []
incorrect_list = []
for i in range(0, y_pred.shape[0]):
    if(np.argmax(y_pred[i]) == 6):
        print("x ", i, " : ", y_pred[i][6], " , ", np.max(y_pred[i]), " , " , np.argmax(y_pred[i]))
        correct_list.append(i)
    else:
        print("x ", i ," : ", y_pred[i][6], " , ", np.max(y_pred[i]), " , " , np.argmax(y_pred[i]))   
        incorrect_list.append(i)

print("Accuracy: ", len(correct_list)/(len(correct_list) + len(incorrect_list)))
print("List of incorrectly classified samples: ", incorrect_list)

for i in range(0, y_pred_adv.shape[0]):
    if(np.argmax(y_pred_adv[i]) != 6):
        print("x ", i, " - ")
        print("map: ", y_pred[i][6], " , ", np.max(y_pred[i]), " , " , np.argmax(y_pred[i]))
        print("adv: ", y_pred_adv[i][6], " , ", np.max(y_pred_adv[i]), " , " , np.argmax(y_pred_adv[i]))
        ax = pyplot.subplot(1, 2, 1)
        plt.imshow(x_map[i,:,:,0], cmap='gray')
        ax = pyplot.subplot(1, 2, 2)
        plt.imshow(x_adv[i,:,:,0], cmap='gray')
        plt.tight_layout()
        plt.show()
        
        #plt.savefig("x_adv_"+str(i)+"_"+str(np.argmax(y_pred[i]))+"_"+str(np.argmax(y_pred_adv[i]))+".png")



# In[ ]:





# In[ ]:




