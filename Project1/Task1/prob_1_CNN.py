import keras
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

'''
batch_size = 128
num_classes = 10
epochs = 12
'''

'''
def baseline_model(num_pixels = 256, num_classes = 10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    #model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer='nadam',metrics=['accuracy'])
    return model
    '''
    
def baseline_model(num_pixels = 256, num_classes = 10,learning_rate=0.001, momentum=0.9):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='glorot_uniform', activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', activation='sigmoid'))
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
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

###Run model

model = baseline_model()
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)
print('CNN Model Test loss:', score[0])
print('CNN Model Test accuracy:', score[1])