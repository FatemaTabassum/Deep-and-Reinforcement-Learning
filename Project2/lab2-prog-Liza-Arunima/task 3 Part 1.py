# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:21:22 2020

@author: Fatema
"""

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
model = load_model('LSTMnewmodel.h5')

tokenizer = Tokenizer(char_level = True)

def generate_text(seed_text, next_words, max_sequence_len):
    predict_text = ''
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
        predict_text += output_word
    return predict_text

def get_number_of_matched_characters(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    count = 0
    for i in range(len(str1)):
        if(str1[i] != str2[i]): 
            break
        else:
            count = count + 1
    return count

def get_number_of_matched_sequences(k, validation_data, length_cutoff, table_row_size):
    print('=====================')
    print('Calculating for k=',k)
    print('=====================')
    counter = 1
    output = [0] * table_row_size
    for sample in validation_data:
        original_seq = sample[k:]
        predicted_seq = generate_text(sample[:k], table_row_size, length_cutoff)
        matched_number = get_number_of_matched_characters(predicted_seq, original_seq)
        print('sample no', counter)
        print('Original Sequence: ', original_seq)
        print('Predicted Seq: ', predicted_seq)
        print('Matched Number: ', matched_number)
        if(matched_number >= table_row_size-1):
            output[table_row_size-1] = output[table_row_size-1] + 1
        else:
            output[matched_number] = output[matched_number] + 1
        counter +=  1
        
    return output



# driver functionality
data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
tokenizer.fit_on_texts(corpus)
print("Total Samples: ", len(corpus))

length_cutoff = 100
table_row_size = 20

# split this into training and validation set
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

for i in range(len(validation_corpus)):
    if(len(validation_corpus[i]) > length_cutoff):
        validation_corpus[i] = validation_corpus[i][:length_cutoff]
    if(len(validation_corpus[i]) < length_cutoff):
       temp = '{:<0' + str(length_cutoff) + '}'
       validation_corpus[i] = temp.format(validation_corpus[i]) 
start_index = 51
end_index = 401
del validation_corpus[start_index:end_index] 
print("Total Samples in Validation Set: ", len(validation_corpus))   
final_output = []    
for k in range(10):
    final_output.append(get_number_of_matched_sequences(k+1, validation_corpus, length_cutoff, table_row_size))

print(final_output)
    