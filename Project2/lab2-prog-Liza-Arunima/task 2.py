from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import matplotlib.pyplot as plt


tokenizer = Tokenizer(char_level = True)

def dataset_preparation(corpus):

    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_amino = len(tokenizer.word_index) + 1
    print(total_amino)
    print("word_index : ",tokenizer.word_index)

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        #print(line)
        #print(token_list)
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    # pad sequences 
    max_sequence_len = 100
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_amino)

    return predictors, label, max_sequence_len, total_amino

data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
training_corpus = []
validation_corpus = []
count = 0
for sample in corpus:
    if count % 5 == 0: 
        validation_corpus.append(sample)
    else: 
        training_corpus.append(sample)
    count = count+1

predictors, label, max_sequence_len, total_amino = dataset_preparation(training_corpus)
y_predictors, y_label, max_sequence, total_amino = dataset_preparation(validation_corpus)

def create_model(predictors, label, max_sequence_len, total_amino):

    model = Sequential()
    model.add(Embedding(total_amino, 24, input_length=max_sequence_len-1))
    model.add(LSTM(256, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(total_amino,kernel_initializer='random_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history = model.fit(predictors, label, epochs=10, verbose=1, validation_data=(y_predictors,y_label))

    print(model.summary())

    #plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model


def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


model = create_model(predictors, label, max_sequence_len, total_amino)
print( generate_text("MVLSEGEWQLVLHVWAK", 3, max_sequence_len))
