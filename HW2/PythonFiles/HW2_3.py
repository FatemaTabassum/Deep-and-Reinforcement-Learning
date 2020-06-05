from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy
import matplotlib.pyplot as plt
import math


def list_generator(num_points=100):
    X_ = numpy.arange(-1.0,1.0, 2/num_points)
    y_ = 1 + numpy.sin(((3*math.pi)/2)*X_)
    return X_, y_

def baseline_model():
    # create model
    model = Sequential()
    adam = keras.optimizers.Adam(lr=0.003)
    
    # Use Activation function tanh
    model.add(Dense(7, input_shape=(1,), activation='tanh'))
    
    # Use Activation function linear
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])
    return model

# Generating list
X, y = list_generator()

# Define the model
model = baseline_model()

print(model.summary())

model.fit(X, y, epochs = 10000, batch_size = 128, verbose = 2)

y_pred = model.predict(X)
diff = numpy.empty(0)
for i in range(0, 100):
    diff = numpy.append(diff, numpy.abs(y[i] - y_pred[i]))
print("Maximum Difference: ",numpy.max(diff))
plt.plot(X, y)
plt.plot(X, y_pred)
plt.show()

