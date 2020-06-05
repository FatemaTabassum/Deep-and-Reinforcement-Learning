from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.utils import resample
from keras.layers.convolutional import Conv2D, MaxPooling2D
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
from keras.utils import np_utils

                                                
    #---------------------- Bagging Ensamble ----------------------#


def load_data(file):
    data = numpy.genfromtxt(file)
    y = data[:,0]
    X = numpy.delete(data,0,axis=1)
    return X,y


# evaluate a fully connected  model
def evaluate_model_dense( train_X, train_y, test_X, test_y, num_pixels = 256, num_classes = 10, learningRate=0.001):
    
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
    # fit model
    model.fit(train_X, train_y_enc, epochs=12, batch_size=12, verbose=2)
    # evaluate the model
    _, test_acc = model.evaluate(test_X, test_y_enc, verbose=0)
    print(test_acc)
    return model, test_acc



# evaluate a conv2D  model
def evaluate_model_conv( train_X, train_y, test_X, test_y, num_pixels = 256, num_classes = 10, learningRate=0.001):

    # encode targets
    train_y_enc = to_categorical(train_y)
    test_y_enc = to_categorical(test_y)
    
    # define model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', input_shape=(16,16,1)))
    #model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    opt = optimizers.nadam(lr=learningRate)

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    # fit model
    model.fit(train_X, train_y_enc, epochs=10, batch_size=8, verbose=2)
    #model.fit(train_X, train_y_enc, epochs=12, batch_size=12, verbose=2)
    
    # evaluate the model
    _, test_acc = model.evaluate(test_X, test_y_enc, verbose=0)
    print(test_acc)
    return model, test_acc




#---------- make an ensemble prediction for multi-class classification --------#

def ensemble_predictions(members, test_X):
    # make predictions
    yhats = [model.predict(test_X) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


#---------- evaluate a specific number of members in an ensemble --------#

def evaluate_n_members(members, n_members, test_X, test_y):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, test_X)
    # calculate accuracy
    return accuracy_score(test_y, yhat)



#------------- generate 2d classification dataset -----------#

X_train, y_train = load_data('train.txt')
X_test, y_test = load_data('test.txt')

new_X = X_test
new_y = y_test

X = X_train
y = y_train


# multiple train-test splits
n_splits = 6

scores, members = list(), list()
scores_conv, members_conv = list(), list()

for _ in range(n_splits):
    # select indexes
    ix = [i for i in range(len(X))]
    train_ix = resample(ix, replace=True, n_samples=6500) # using 90%of training data for resampling and bagging
    test_ix = [x for x in ix if x not in train_ix]
    # select data
    train_X, train_y = X[train_ix], y[train_ix]
    test_X, test_y = X[test_ix], y[test_ix]
    
    # evaluate model dense
    #'''    
    model, test_acc = evaluate_model_dense(train_X, train_y, test_X, test_y)
    print('dense >%.3f' % test_acc)
    scores.append(test_acc)
    members.append(model)
    #'''    
    print(test_X.shape[0])
    
    # evaluate model conv
    X_train_conv = train_X.reshape(train_X.shape[0],16,16,1)
    X_test_conv = test_X.reshape(test_X.shape[0],16,16,1)
    print(X_test_conv.shape[0])

    # evaluate model conv
    model_conv, test_acc_conv = evaluate_model_conv(X_train_conv, train_y, X_test_conv, test_y)
    print('conv >%.3f' % test_acc_conv)
    scores_conv.append(test_acc_conv)
    members_conv.append(model_conv)
    
    

    
# summarize expected performance dense
#'''    
print('Estimated Accuracy dense %.3f (%.3f)' % (mean(scores), std(scores)))
#'''    

# summarize expected performance conv
print('Estimated Accuracy conv %.3f (%.3f)' % (mean(scores_conv), std(scores_conv)))

print
# evaluate different numbers of ensembles on hold out set : Dense
#'''
single_scores, ensemble_scores = list(), list()
for i in range(1, n_splits+1):
    ensemble_score = evaluate_n_members(members, i, new_X, new_y)
    new_y_enc = to_categorical(new_y)
    _, single_score = members[i-1].evaluate(new_X, new_y_enc, verbose=0)
    print('Dense > %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
    ensemble_scores.append(ensemble_score)
    single_scores.append(single_score)
    
#'''    
print()    
# evaluate different numbers of ensembles on hold out set : Conv
single_scores_conv, ensemble_scores_conv = list(), list()
for i in range(1, n_splits+1):
    new_y_enc = to_categorical(new_y)
    new_X = new_X.reshape(new_X.shape[0],16,16,1)  
    ensemble_score = evaluate_n_members(members_conv, i, new_X, new_y) 
    #X_test_conv = test_X.reshape(test_X.shape[0],16,16,1)
    _, single_score = members_conv[i-1].evaluate(new_X, new_y_enc, verbose=0)
    print('Conv > %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
    single_scores_conv.append(single_score)
    ensemble_scores_conv.append(ensemble_score)
    
    
    
# plot score vs number of ensemble members : dense
#'''   
print()
print('Accuracy dense %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
#'''    

# plot score vs number of ensemble members : conv
print()
print('Accuracy Conv %.3f (%.3f)' % (mean(single_scores_conv), std(single_scores_conv)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores_conv, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores_conv, marker='o')
pyplot.show()


