import os
import sys
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Input,Activation,Dense,Dropout,LSTM
from keras.models import Model

data = open ('data.txt','r')
count = 0
MAX_SEQUENCE_LENGTH = 0
samples = []           
categories = []        
categories_sl = {}     
for line in data:
    samples.append(line[line.find(' ')+1:-2])
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, len(line[line.find(' ')+1:-2].split(' ')))
    categories.append(line[:line.find(' ')])
    if line[:line.find(' ')] not in categories_sl.keys():
        categories_sl[line[:line.find(' ')]] = count
        count +=1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
array = np.asarray([categories_sl[category] for category in categories])
categories = np_utils.to_categorical(array)

embeddings_index = {}
f = open('glove.6B.200d.txt','r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
categories = categories[indices]
nb_validation_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = categories[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = categories[-nb_validation_samples:]

embedding_layer = Embedding(len(word_index) + 1,200,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(len(categories_sl), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])


model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=50)
