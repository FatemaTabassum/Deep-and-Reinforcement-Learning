import keras
import matplotlib.pyplot as plt
from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import np_utils

def data_loader(file):
	data = np.genfromtxt(file)
	y = data[:,0]
	X = np.delete(data,0,axis=1)
	return X,y

    
def learn_effec_model(num_pixels = 256, num_classes = 10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='glorot_uniform', activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer='nadam',metrics=['accuracy'])
    return model

def learn_slow_model(num_pixels = 256, num_classes = 10, initializer_val=100000):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer=initializers.Constant(value=initializer_val), activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer=initializers.Constant(value=initializer_val), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=initializers.Constant(value=initializer_val), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer=initializers.Constant(value=initializer_val), activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer='nadam',metrics=['accuracy'])
    return model
    
def learn_fast_model(num_pixels = 256, num_classes = 10, initializer_val=0.01):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer=initializers.Constant(value=initializer_val), activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer=initializers.Constant(value=initializer_val), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=initializers.Constant(value=initializer_val), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer=initializers.Constant(value=initializer_val), activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer='nadam',metrics=['accuracy'])
    return model

### Prepare data

X_train, y_train = data_loader('train.txt')
X_test, y_test = data_loader('test.txt')
X_train = X_train.reshape(7291,16,16,1)
X_test = X_test.reshape(2007,16,16,1)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


###Run Models

model = learn_effec_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_effec = history.history['loss']
test_loss_effec = history.history['val_loss']
train_acc_effec = history.history['accuracy']
test_acc_effec = history.history['val_accuracy']

model = learn_slow_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_slow = history.history['loss']
test_loss_slow = history.history['val_loss']
train_acc_slow = history.history['accuracy']
test_acc_slow = history.history['val_accuracy']

model = learn_fast_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_fast = history.history['loss']
test_loss_fast = history.history['val_loss']
train_acc_fast = history.history['accuracy']
test_acc_fast = history.history['val_accuracy']



print('###Printing model accuracy###')

print('train_acc_effec',train_acc_effec)
print('train_acc_slow',train_acc_slow)
print('train_acc_fast',train_acc_fast)

print('test_acc_effec',test_acc_effec)
print('test_acc_slow',test_acc_slow)
print('test_acc_fast',test_acc_fast)

print('###Printing model loss###')

print('train_loss_effec',train_loss_effec)
print('train_loss_slow',train_loss_slow)
print('train_loss_fast',train_loss_fast)

print('test_loss_effec', test_loss_effec)
print('test_loss_slow',test_loss_slow)
print('test_loss_fast',test_loss_fast)

### Plotting for model accuracy
plt.plot(train_acc_effec)
plt.plot(train_acc_slow)
plt.plot(train_acc_fast)

plt.plot(test_acc_effec)
plt.plot(test_acc_slow)
plt.plot(test_acc_fast)

plt.title('CNN model accuracy with varied initializer')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc_effec','train_acc_slow','train_acc_fast','test_acc_effec','test_acc_slow','test_acc_fast'], loc='center')
plt.show()

### Plotting for model loss
plt.plot(train_loss_effec)
plt.plot(train_loss_slow)
plt.plot(train_loss_fast)

plt.plot(test_loss_effec)
plt.plot(test_loss_slow)
plt.plot(test_loss_fast)

plt.title('CNN model loss with varied initializer')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss_effec','train_loss_slow','train_loss_fast','test_loss_effec','test_loss_slow','test_loss_fast'], loc='center')
plt.show()