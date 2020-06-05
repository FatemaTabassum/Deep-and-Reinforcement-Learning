import numpy
import matplotlib.pyplot as plt
from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

def data_loader(file):
	data = numpy.genfromtxt(file)
	y = data[:,0]
	X = numpy.delete(data,0,axis=1)
	return X,y

    
def learn_effec_model_mom1(num_pixels = 256, num_classes = 10,learning_rate=0.001, momentum=0.5):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', activation='sigmoid'))
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
def learn_effec_model_mom2(num_pixels = 256, num_classes = 10,learning_rate=0.001, momentum=0.9):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', activation='sigmoid'))
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def learn_effec_model_mom3(num_pixels = 256, num_classes = 10,learning_rate=0.001, momentum=0.99):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='elu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', activation='sigmoid'))
    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

X_train, y_train = data_loader('train.txt')
X_test, y_test = data_loader('test.txt')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



model = learn_effec_model_mom1()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_mom1 = history.history['loss']
test_loss_mom1 = history.history['val_loss']
train_acc_mom1 = history.history['accuracy']
test_acc_mom1 = history.history['val_accuracy']

model = learn_effec_model_mom2()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_mom2 = history.history['loss']
test_loss_mom2 = history.history['val_loss']
train_acc_mom2 = history.history['accuracy']
test_acc_mom2 = history.history['val_accuracy']

model = learn_effec_model_mom3()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_mom3 = history.history['loss']
test_loss_mom3 = history.history['val_loss']
train_acc_mom3 = history.history['accuracy']
test_acc_mom3 = history.history['val_accuracy']


print('###Printing model accuracy###')

print('train_acc_mom1',train_acc_mom1)
print('train_acc_mom2',train_acc_mom2)
print('train_acc_mom3',train_acc_mom3)

print('test_acc_mom1',test_acc_mom1)
print('test_acc_mom2',test_acc_mom2)
print('test_acc_mom3',test_acc_mom3)


print('###Printing model loss###')

print('train_loss_mom1',train_loss_mom1)
print('train_loss_mom2',train_loss_mom2)
print('train_loss_mom3',train_loss_mom3)

print('test_loss_mom1', test_loss_mom1)
print('test_loss_mom2',test_loss_mom2)
print('test_loss_mom3',test_loss_mom3)



### Plotting for model accuracy
plt.plot(train_acc_mom1)
plt.plot(train_acc_mom2)
plt.plot(train_acc_mom3)

plt.plot(test_acc_mom1)
plt.plot(test_acc_mom2)
plt.plot(test_acc_mom3)

plt.title('Fully-connected model accuracy with varied momentum')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc_mom1','train_acc_mom2','train_acc_mom3','test_acc_mom1','test_acc_mom2','test_acc_mom3'], loc='lower right')
plt.show()

### Plotting for model loss
plt.plot(train_loss_mom1)
plt.plot(train_loss_mom2)
plt.plot(train_loss_mom3)

plt.plot(test_loss_mom1)
plt.plot(test_loss_mom2)
plt.plot(test_loss_mom3)

plt.title('Fully-connected model loss with varied momentum')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss_mom1','train_loss_mom2','train_loss_mom3','test_loss_mom1','test_loss_mom2','test_loss_mom3'], loc='upper right')
plt.show()
