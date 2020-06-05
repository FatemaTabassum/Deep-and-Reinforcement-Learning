import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras import optimizers
from keras.utils import np_utils
import copy
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_data(file):
    data = np.genfromtxt(file)
    y = data[:,0]
    X = np.delete(data,0,axis=1)
    return X,y


# evaluate a fully connected  model
def evaluate_model_dense_1( train_X, train_y, test_X, test_y, num_pixels = 256, num_classes = 10, learningRate=0.001):
    
    # encode targets
    train_y_enc = to_categorical(train_y)
    test_y_enc = to_categorical(test_y)
    
    #have to replace with our models
    # define model
    model = Sequential()
    
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='elu'))
    model.add(Dense(128, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    
    opt = optimizers.nadam(lr=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# evaluate a fully connected  model
def evaluate_model_dense_2( train_X, train_y, test_X, test_y, num_pixels = 256, num_classes = 10, learningRate=0.001):
    
    # encode targets
    train_y_enc = to_categorical(train_y)
    test_y_enc = to_categorical(test_y)
    
    #have to replace with our models
    # define model
    model = Sequential()
    
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='elu'))
    model.add(Dense(128, kernel_initializer='normal', activation='elu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    
    opt = optimizers.nadam(lr=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def baseline_model_conv_1( train_X, train_y, test_X, test_y,  num_pixels = 256, num_classes = 10, learningRate=0.001, ):
    
     # encode targets
    train_y_enc = to_categorical(train_y)
    test_y_enc = to_categorical(test_y)
    
    # define model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', input_shape=(16,16,1)))
    #model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    opt = optimizers.nadam(lr=learningRate)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    return model

def baseline_model_conv_2( train_X, train_y, test_X, test_y,  num_pixels = 256, num_classes = 10, learningRate=0.001, ):
    
     # encode targets
    train_y_enc = to_categorical(train_y)
    test_y_enc = to_categorical(test_y)
    
    # define model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', input_shape=(16,16,1)))
    model.add(Dropout(0.25))

    #model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    opt = optimizers.nadam(lr=learningRate)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    return model



# load data
X_train, y_train = load_data('train.txt')
X_test, y_test = load_data('test.txt')

                    #---------------------- Dense -------------#

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#without dense
model = evaluate_model_dense_1( X_train, y_train, X_test, y_test)
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)
train_loss_effec = history.history['loss']
test_loss_effec = history.history['val_loss']
train_acc_effec = history.history['acc']
test_acc_effec = history.history['val_acc']

#with dense
model = evaluate_model_dense_2( X_train, y_train, X_test, y_test)
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)
train_loss_ineffec = history.history['loss']
test_loss_ineffec = history.history['val_loss']
train_acc_ineffec = history.history['acc']
test_acc_ineffec = history.history['val_acc']

print('###Printing model accuracy###')

print('train_acc_effec',train_acc_effec)
print('train_acc_ineffec',train_acc_ineffec)

print('test_acc_effec',test_acc_effec)
print('test_acc_ineffec',test_acc_ineffec)

print('###Printing model loss###')

print('train_loss_effec',train_loss_effec)
print('train_loss_ineffec',train_loss_ineffec)

print('test_loss_effec', test_loss_effec)
print('test_loss_ineffec',test_loss_ineffec)


print('### Printing model accuracy ###')

print('train_acc_effec',train_acc_effec)
print('train_acc_ineffec',train_acc_ineffec)

print('test_acc_effec',test_acc_effec)
print('test_acc_ineffec',test_acc_ineffec)

print('###Printing model loss###')

print('train_loss_effec',train_loss_effec)
print('train_loss_ineffec',train_loss_ineffec)

print('test_loss_effec', test_loss_effec)
print('test_loss_ineffec',test_loss_ineffec)


### Plotting for model accuracy
plt.plot(train_acc_effec)
plt.plot(train_acc_ineffec)

plt.plot(test_acc_effec)
plt.plot(test_acc_ineffec)

plt.title('Dense model accuracy - Dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc_effec','train_acc_ineffec','test_acc_effec','test_acc_ineffec'], loc='lower right')
plt.show()

### Plotting for model loss
plt.plot(train_loss_effec)
plt.plot(train_loss_ineffec)

plt.plot(test_loss_effec)
plt.plot(test_loss_ineffec)

plt.title('Dense model loss - Dropout')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss_effec','train_loss_ineffec','test_loss_effec','test_loss_ineffec'], loc='upper right')
plt.show()



                    #---------------------- Conv2D -------------#

# reshape to 2D array
X_train = X_train.reshape(7291,16,16,1)
X_test = X_test.reshape(2007,16,16,1)


y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


model = baseline_model_conv_1( X_train, y_train, X_test, y_test, num_pixels = 256, num_classes = 10)
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)
train_loss_effec = history.history['loss']
test_loss_effec = history.history['val_loss']
train_acc_effec = history.history['acc']
test_acc_effec = history.history['val_acc']

model = baseline_model_conv_2( X_train, y_train, X_test, y_test, num_pixels = 256, num_classes = 10)
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)
train_loss_ineffec = history.history['loss']
test_loss_ineffec = history.history['val_loss']
train_acc_ineffec = history.history['acc']
test_acc_ineffec = history.history['val_acc']



print('### Printing model accuracy ###')

print('train_acc_effec',train_acc_effec)
print('train_acc_ineffec',train_acc_ineffec)

print('test_acc_effec',test_acc_effec)
print('test_acc_ineffec',test_acc_ineffec)

print('###Printing model loss###')

print('train_loss_effec',train_loss_effec)
print('train_loss_ineffec',train_loss_ineffec)

print('test_loss_effec', test_loss_effec)
print('test_loss_ineffec',test_loss_ineffec)


### Plotting for model accuracy
plt.plot(train_acc_effec)
plt.plot(train_acc_ineffec)

plt.plot(test_acc_effec)
plt.plot(test_acc_ineffec)

plt.title('Conv2D model accuracy - Dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc_effec','train_acc_ineffec','test_acc_effec','test_acc_ineffec'], loc='lower right')
plt.show()

### Plotting for model loss
plt.plot(train_loss_effec)
plt.plot(train_loss_ineffec)

plt.plot(test_loss_effec)
plt.plot(test_loss_ineffec)

plt.title('Conv2D model loss - Dropout')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss_effec','train_loss_ineffec','test_loss_effec','test_loss_ineffec'], loc='upper right')
plt.show()



