import numpy

bias = numpy.load("softmax_bias.npy")
Weights = numpy.load("softmax_weights.npy")

Weights = numpy.reshape(Weights, [100, 20])
bias = numpy.reshape(bias, [20, 1])

asample = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 1.606391429901123, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.9543248414993286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.1392189860343933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.836493968963623, 0.0, 
                    0.12610933184623718, 0.0, 0.0, 0.0, 0.0843304991722107,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4557386338710785, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.3026450276374817, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.6092420816421509, 0.23424609005451202, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0]);
asample = numpy.reshape(asample, [100, 1])

# Transpose Weights matrix so that the values of each neuron could be found
Weights = Weights.T

# Compute the value of each neurons
neurons_value = numpy.dot(Weights, asample) + bias

#Predict output now
y_prdict = numpy.exp(neurons_value) / numpy.sum(numpy.exp(neurons_value))
print(y_prdict)

print("Prediction : ", numpy.argmax(y_prdict) + 1)

# Assuming Class label 1
class_label = 1

# Cross entropy
loss = -numpy.log(y_prdict[class_label-1])
print(loss)

#Compute Gradient Descent for Weights

gdw = -(y_prdict[class_label-1] - y_prdict[class_label-1]*y_prdict[class_label-1])*asample/y_prdict[class_label-1]

#### QS_ANS_PART_3 ####
#Compute Gradient Descent for Bias

gdb = -(y_prdict[class_label-1] -
            y_prdict[class_label-1]*y_prdict[class_label-1])/y_prdict[class_label-1]

learning_rate = 0.1

Weights_cpy = numpy.copy(Weights)
bias_cpy = numpy.copy(bias)

Weights_cpy[class_label-1] -= learning_rate*gdw.ravel()
bias_cpy[class_label-1] -= learning_rate*gdb.ravel()

W_diff = Weights_cpy - Weights
B_diff = bias_cpy - bias

increase_w = 0
decrease_w = 0
increase_b = 0
decrease_b = 0
no_change_w = 0
no_change_b = 0

for e in W_diff.ravel():
    if(e > 0.001):
        increase_w += 1
    else:
        if(e < -0.001):
            decrease_w += 1
        else:
            no_change_w += 1

for e in B_diff.ravel():
    if(e > 0.001):
        increase_b += 1
    else:
        if(e < -0.001):
            decrease_b += 1
        else:
            no_change_b += 1 
print("Number of Weights Increased",increase_w)
print("Number of Weights decreased", decrease_w)
print("Number of Weights Unchanged", no_change_w)
print("\n")
print("Number of Biases Increased",increase_b)
print("Number of Biases decreased", decrease_b)
print("Number of Biases Unchanged", no_change_b)


#### QS_ANS_PART_4 ####

inverting_one =numpy.matmul(Weights, Weights.T)

inverted_one = numpy.linalg.inv(inverting_one)

B = numpy.zeros(20)
val = numpy.dot(Weights,asample)+ bias
B[1] = val[numpy.argmax(y_prdict)] - val[1]

del_x = numpy.matmul(numpy.matmul(Weights.T, inverted_one), B)
del_x = numpy.reshape(del_x, [100,1])

asample = asample + del_x

# predict
neurons_value = numpy.dot(Weights, asample) + bias
y_prdict = numpy.exp(neurons_value) / numpy.sum(numpy.exp(neurons_value))



print("Adverserial Model Prediction: ", numpy.argmax(y_prdict) + 1)