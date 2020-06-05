import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

def data_loader(file):
	data = numpy.genfromtxt(file)
	y = data[:,0]
	X = numpy.delete(data,0,axis=1)
	return X,y

def baseline_model(num_pixels = 256, num_classes = 10, kernel_initializer='glorot_uniform'):
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer=kernel_initializer, activation='elu'))
	model.add(Dense(128, kernel_initializer=kernel_initializer, activation='elu'))
	model.add(Dense(64, kernel_initializer=kernel_initializer, activation='relu'))
	model.add(Dense(32, kernel_initializer=kernel_initializer, activation='relu'))
	model.add(Dense(16, kernel_initializer=kernel_initializer, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer=kernel_initializer, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
	return model

X_train, y_train = data_loader('train.txt')
X_test, y_test = data_loader('test.txt')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
model = baseline_model()
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)
print('Fully-Connected network Test loss:', score[0])
print('Fully-Connected network Test accuracy:', score[1])