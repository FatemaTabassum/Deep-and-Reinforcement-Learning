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


def learn_effec_model(num_pixels = 256, num_classes = 10, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='elu'))
    model.add(Dense(128, kernel_initializer='normal', activation='elu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    optimizer = optimizers.nadam(lr=learning_rate)
    #optimizer = optimizers.SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


X_train, y_train = data_loader('train.txt')
X_test, y_test = data_loader('test.txt')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



model = learn_effec_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)
train_loss_effec = history.history['loss']
test_loss_effec = history.history['val_loss']
train_acc_effec = history.history['accuracy']
test_acc_effec = history.history['val_accuracy']

model = learn_effec_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=1024, verbose=1)
train_loss_ineffec = history.history['loss']
test_loss_ineffec = history.history['val_loss']
train_acc_ineffec = history.history['accuracy']
test_acc_ineffec = history.history['val_accuracy']


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


### Plotting for model accuracy
plt.plot(train_acc_effec)
plt.plot(train_acc_ineffec)

plt.plot(test_acc_effec)
plt.plot(test_acc_ineffec)

plt.title('Fully-connected model accuracy with varied batch size')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc_effec','train_acc_ineffec','test_acc_effec','test_acc_ineffec'], loc='lower right')
plt.show()


### Plotting for model loss
plt.plot(train_loss_effec)
plt.plot(train_loss_ineffec)

plt.plot(test_loss_effec)
plt.plot(test_loss_ineffec)

plt.title('Fully-connected model loss with varied batch size')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss_effec','train_loss_ineffec','test_loss_effec','test_loss_ineffec'], loc='upper right')
plt.show()
