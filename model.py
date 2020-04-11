print("\nLoading the app....")
print("Please wait....\n")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import urllib.request as request
from os import path
import os

def download_url(url, save_path):
    with request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())

saved_model = ['https://github.com/sankirnajoshi/sentiment-app/raw/master/model/model.h5',
              'https://raw.githubusercontent.com/sankirnajoshi/sentiment-app/master/model/model.json',
              'https://github.com/sankirnajoshi/sentiment-app/raw/master/model/tokenizer.pickle'
              ]

if not path.exists('./model'):
    os.makedirs('./model')

download_url(saved_model[0],'./model/model.h5')
download_url(saved_model[1],'./model/model.json')
download_url(saved_model[2],'./model/tokenizer.pickle')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing import sequence
import numpy as np
import pickle

with open('../data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load json and create model
json_file = open('../data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../data/model.h5")

def predict_response(response):
    response = tokenizer.texts_to_sequences(response)
    response = sequence.pad_sequences(response, maxlen=48)
    pred = np.argmax(model.predict(response))
    if pred == 0:
        return 'Very Negative'
    elif pred == 1:
        return 'Somewhat Negative'
    elif pred == 2:
        return 'Nuetral'
    elif pred == 3:
        return 'Somewhat Positive'
    elif pred == 4:
        return 'Very Positive'

print("Loading Complete...\n\n\n\n")
print("SENTIMENT ANALYSIS APPLICATION\n")

while True:             
    user_input = input("Please enter your text below:\n")
    if user_input == "":       
        print("Thank you.")
        break
    response = predict_response(response=[user_input])
    print(f'\nPredicted Sentiments are:\n{response}\n')


