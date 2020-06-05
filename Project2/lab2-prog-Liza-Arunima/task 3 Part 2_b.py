# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:34:32 2020

@author: Fatema
"""
import operator

sequence_frequency_map = {}


# function to generate n_gram
def generate_n_gram(sample, n):
    count = n
    sum = 0;
    for i in range(len(sample)-n+1):
        n_gram = sample[i:count]
        if(sequence_frequency_map.get(n_gram) == None):
            sequence_frequency_map[n_gram] = 1
        else:
            sequence_frequency_map[n_gram] = sequence_frequency_map[n_gram] + 1
        count = count+1
        sum = sum + 1
    return sum
        
# constant declaration      
length_cutoff = 100
n = 4

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

# trim the data
for i in range(len(training_corpus)):
    if(len(training_corpus[i]) > length_cutoff):
        training_corpus[i] = training_corpus[i][:length_cutoff]


# generate n_gram
sample_count = 1
tot_sum = 0
for sample in training_corpus:
    print('======================')
    print('Sample #', sample_count, ' ', sample)
    tot_sum = tot_sum + generate_n_gram(sample, n)
    print('Size of the 4-gram set: ', len(sequence_frequency_map))
    sample_count = sample_count+1
    print('======================')

four_gram_model = sequence_frequency_map
print('tot_sum = ', tot_sum);
for key, value in four_gram_model.items():
    four_gram_model[key] = value/tot_sum
    
print('Probabilities of all the 4-gram sequences based on trained model')
print('================================================================')
for key, value in four_gram_model.items():
   print(key, '->', value)
    
   
    
       