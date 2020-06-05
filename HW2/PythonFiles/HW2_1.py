import numpy
import matplotlib.pyplot as plt 




def ReLU(X):
    y = [1 if (x > 0) else 0 for x in X]
    y = numpy.array(y)
    return y



def sigmoid(X):
    fx = 1/(1 + numpy.exp(-X))
    y = fx*(1 - fx)
    return y


def piece_wise_linearUnit(X):
    y = [(0.05) if x > 1 else (0.05) if x < -1 else 1 for x in X]
    y = numpy.array(y)
    return y



def swish(X): 
    fx = 1/(1 + numpy.exp(-5*X))
    
    # Finding Gradient Descent according to the definition of Swish
    y = fx + 5*X*numpy.exp(-5*X)*fx*fx
    return y



def ELU(X):
    # Finding GD according to the definition of ELU
    a = 0.5
    y = [1 if (x >= 0) else a*numpy.exp(x) for x in X]
    return np.array(y)    

X = np.arange(-10,11,0.001)



X = numpy.arange(-10,11,0.001)


#SWISH
y = swish(X)

fast_learning_region = numpy.where(numpy.abs(y) > 0.9)[0]
active_learning_region = numpy.where((numpy.abs(y) >= 0.1) & (numpy.abs(y) <= 0.9))[0]
slow_learning_region = numpy.where(numpy.abs(y) < 0.1)[0]
inactive_learning_region = numpy.where(numpy.abs(y) == 0)[0]


plt.plot(X[slow_learning_region], y[slow_learning_region], color='b', label='Slow Learning Region')
plt.plot(X[active_learning_region], y[active_learning_region], color='g', label='Active Learning Region')
plt.plot(X[fast_learning_region], y[fast_learning_region], color='r', label='Fast Learning Region')
plt.plot(X[inactive_learning_region], y[inactive_learning_region], color='m', label='Inactive Learning Region')

plt.ylabel("f'(z)")
plt.xlabel("z")
plt.title("Learning Region of Swish")
plt.show()


# SIGMOID
y = sigmoid(X)

fast_learning_region = numpy.where(y > 0.9)[0]
active_learning_region = numpy.where((y >= 0.1) & (y <= 0.9))[0]
slow_learning_region_positive = numpy.where(y < 0.1)[0]
inactive_learning_region = numpy.where(y == 0.0)[0]

plt.plot(X[slow_learning_region], y[slow_learning_region], color='b', label='Slow Learning Region')
plt.plot(X[active_learning_region], y[active_learning_region], color='g', label='Active Learning Region')
plt.plot(X[fast_learning_region], y[fast_learning_region], color='r', label='Fast Learning Region')
plt.plot(X[inactive_learning_region], y[inactive_learning_region], color='m', label='Inactive Learning Region')

plt.ylabel("f'(z)")
plt.xlabel("z")
plt.title("Learning Region of Sigmoid")
plt.legend()
plt.show()


# PIECE_WISE_LINEAR_UNIT
y = piece_wise_linearUnit(X)

fast_learning_region = numpy.where(y > 0.9)[0]
active_learning_region = numpy.where((y >= 0.1) & (y <= 0.9))[0]
slow_learning_region = numpy.where(y < 0.1)[0]
inactive_learning_region = numpy.where(y == 0.0)[0]

plt.plot(X[slow_learning_region], y[slow_learning_region], color='b', label='Slow Learning Region')
plt.plot(X[active_learning_region], y[active_learning_region], color='g', label='Active Learning Region')
plt.plot(X[fast_learning_region], y[fast_learning_region], color='r', label='Fast Learning Region')
plt.plot(X[inactive_learning_region], y[inactive_learning_region], color='m', label='Inactive Learning Region')

plt.ylabel("f'(z)")
plt.xlabel("z")
plt.title("Learning Region of Piece Wise Linear Unit")
plt.legend()
plt.show()

# ReLU
y = ReLU(X)

fast_learning_region = numpy.where(y > 0.9)[0]
active_learning_region = numpy.where((y >= 0.1) & (y <= 0.9))[0]
slow_learning_region = numpy.where(y < 0.1)[0]
inactive_learning_region = numpy.where(y == 0.0)[0]

plt.plot(X[slow_learning_region], y[slow_learning_region], color='b', label='Slow Learning Region')
plt.plot(X[active_learning_region], y[active_learning_region], color='g', label='Active Learning Region')
plt.plot(X[fast_learning_region], y[fast_learning_region], color='r', label='Fast Learning Region')
plt.plot(X[inactive_learning_region], y[inactive_learning_region], color='m', label='Inactive Learning Region')

plt.ylabel("f'(z)")
plt.xlabel("z")
plt.title("Learning Region of ReLU")
plt.legend()
plt.show()


# ELU
y = ELU(X)

fast_learning_region = numpy.where(y > 0.9)[0]
active_learning_region = numpy.where((y >= 0.1) & (y <= 0.9))[0]
slow_learning_region = numpy.where(y < 0.1)[0]
inactive_learning_region = numpy.where(y == 0.0)[0]

plt.plot(X[slow_learning_region], y[slow_learning_region], color='b', label='Slow Learning Region')
plt.plot(X[active_learning_region], y[active_learning_region], color='g', label='Active Learning Region')
plt.plot(X[fast_learning_region], y[fast_learning_region], color='r', label='Fast Learning Region')
plt.plot(X[inactive_learning_region], y[inactive_learning_region], color='m', label='Inactive Learning Region')

plt.ylabel("f'(z)")
plt.xlabel("z")
plt.title("Learning Region of ELU")
plt.legend()
plt.show()