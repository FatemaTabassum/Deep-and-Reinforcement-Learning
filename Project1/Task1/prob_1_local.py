import numpy
from keras.models import Sequential
from keras.layers import Dense, Flatten, LocallyConnected1D
from keras.utils import np_utils

def data_loader(file):
	data = numpy.genfromtxt(file)
	y = data[:,0]
	X = numpy.delete(data,0,axis=1)
    #print (X.shape)
	X = X.reshape(X.shape[0], X.shape[1], 1)
	#print(X.shape)
	return X,y

def baseline_model(num_pixels = 256, num_classes = 10):
    model = Sequential()
    model.add(LocallyConnected1D(num_pixels, 3, input_shape=(num_pixels,1), kernel_initializer='normal', activation='elu'))
    model.add(LocallyConnected1D(128, 3, kernel_initializer='normal', activation='elu'))
    model.add(LocallyConnected1D(64, 3, kernel_initializer='normal', activation='elu'))
    model.add(LocallyConnected1D(32, 3, kernel_initializer='normal', activation='elu'))
    model.add(LocallyConnected1D(16, 3, kernel_initializer='normal', activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

X_train, y_train = data_loader('train.txt')
X_test, y_test = data_loader('test.txt')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
model = baseline_model()
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Locally Connected Test loss:', score[0])
print('Locally Connected Test accuracy:', score[1])