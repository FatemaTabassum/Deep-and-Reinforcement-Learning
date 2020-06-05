# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:16:34 2020

@author: arunima
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

tokenizer = Tokenizer(char_level = True)
model = load_model('LSTMnewmodel.h5', custom_objects=None, compile=True)  
length_cutoff = 100
n = 4

n_gram_probability_map = {}

def calculate_word_probability(seed_text, word):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=99, padding='pre')
    predicted_probabilities = model.predict(token_list, verbose=0)
    return predicted_probabilities[0][tokenizer.word_index[word]]
    

def calculate_probability(n_gram):
    output_probability = 1
    for i in range(len(n_gram)):
        output_probability = output_probability*calculate_word_probability(n_gram[:i], n_gram[i:i+1])
    return output_probability

# function to generate n_gram
def generate_n_gram(sample, n):
    count = n
    for i in range(len(sample)-n+1):
        n_gram = sample[i:count]
        if(n_gram_probability_map.get(n_gram) == None):
            probability = calculate_probability(n_gram)
            n_gram_probability_map[n_gram] = probability
        count = count+1

# driver functionality 
data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
print("Total Samples: ", len(corpus))

# split the data into training and validation set
training_corpus = []
validation_corpus = []
count = 0
for sample in corpus:
    if count % 5 == 0: 
        validation_corpus.append(sample)
    else: 
        training_corpus.append(sample)
    count = count+1

print("Total Samples in Training Set: ", len(training_corpus))
print("Total Samples in Validation Set: ", len(validation_corpus))

# fit the tokenizer
tokenizer.fit_on_texts(training_corpus)

# trim the data
for i in range(len(training_corpus)):
    if(len(training_corpus[i]) > length_cutoff):
        training_corpus[i] = training_corpus[i][:length_cutoff]

# generate n_gram
sample_count = 1
for sample in training_corpus:
    print('======================')
    print('Sample #', sample_count, ' ', sample)
    generate_n_gram(sample.lower(), n)
    print('Size of the 4-gram set: ', len(n_gram_probability_map))
    sample_count = sample_count+1
    print('======================')

print('Probabilities of all the 4-gram sequences based on trained model')
print('================================================================')
for key, value in n_gram_probability_map.items():
    print(key, '->', value)