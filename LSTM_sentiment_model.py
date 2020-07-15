import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Activation
from keras.layers.embeddings import Embedding
import os
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

max_words = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(seed=1, num_words=max_words)

max_review_len=80
X_train=pad_sequences(X_train,truncating='pre', padding='pre', maxlen=max_review_len)
X_test=pad_sequences(X_test,truncating='pre', padding='pre', maxlen=max_review_len)
print(len(X_train[0]))

embed_vector_len=32

model=Sequential()
model.add(Embedding(input_dim=max_words,output_dim=embed_vector_len,embeddings_initializer='uniform',mask_zero=True))
model.add(LSTM(100,kernel_initializer='uniform',dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

History=model.fit(X_train,y_train,shuffle=True,batch_size=128,epochs=5)

score=model.evaluate(X_test,y_test)
print(score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print(score)

#To test the model with individual test data
d=imdb.get_word_index()
review=input().lower()
words = review.split()
review = []
for word in words:
    if word in d:
        if d[word] > 20000: 
            review.append(2)
        else:
            review.append(d[word]+3)
    else:
        review.append(2)

review=pad_sequences([review],truncating='pre', padding='pre', maxlen=max_review_len)
#print(review)
prediction=model.predict(review)
print(prediction)
if(prediction[0][0]<0.5):
  print('Negetive')
else:
  print('Positive')