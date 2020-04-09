import urllib.request
import sys
import zipfile
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import  stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from os import path

import warnings
warnings.filterwarnings("ignore")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')


stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

url = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'

def download_url(url, save_path):
    '''
    Download the zip file from stanford website
    '''
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())


def dataset_preparation(filename, header=None):
    '''
    Extract the downloaded zip with the folder structure intact. 
    ##To Do - optimize the function to remove folder structure and download only the required files.
    '''
    data = pd.read_csv(filename, sep="|", header=header)
    return data    


def target_map(target):
    '''
    Apply function to target to get ratings in 
    desired form. Ratings will be converted from 
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    to [0,1,2,3,4]. The new ratings are defined as :
    0 - very negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - very positive
    '''
    if 0 <= target <= 0.2:
        return 0
    elif 0.2 < target <=0.4:
        return 1
    elif 0.4 < target <=0.6:
        return 2
    elif 0.6 < target <=0.8:
        return 3
    else:
        return 4

def clean_data(df, col):
    ##Remove records without any words
    df = df[df[col].str.contains('[A-Za-z]')] #remove rows without any alphabets.
    df[col] = df[col].str.replace('[^\w\s]','').str.lower().str.strip() #replace punctuations, remove case and strip whitespaces
    df = df.drop_duplicates(subset=[col])
    df[col] = [BeautifulSoup(text).get_text() for text in df[col]] #handling any HTML tags in the input
    df[col] = df[col].apply(lambda x: word_tokenize(x)) #converting sentences to list of words or tokens
    df[col] = df[col].apply(lambda x: [item for item in x if item not in stop_words]) #removing stop words like 'a', 'the', 'on' etc
    df[col] = df[col].apply(lambda x : [lemmatizer.lemmatize(item) for item in x]) #get word lemmas. eg.  lemma of 'is','are','were' is 'be' etc.
    return df

if not path.exists('../data'): #avoid downloading data everytime
    download_url(url,'../data.zip')
    with zipfile.ZipFile('../data.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/')

dictionary = dataset_preparation('../data/stanfordSentimentTreebank/dictionary.txt')
dictionary.columns = ['phrase','phrase_id']

sentiment = dataset_preparation('../data/stanfordSentimentTreebank/sentiment_labels.txt', header=0)
sentiment.columns=['phrase_id','target']

data = pd.merge(dictionary, sentiment, how='inner',on='phrase_id')

data['target'] = data['target'].map(target_map)
data = clean_data(data,'phrase')
data = data.set_index('phrase_id')


##Modeling

from sklearn.model_selection import train_test_split
X = data.drop('target', axis = 1)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

