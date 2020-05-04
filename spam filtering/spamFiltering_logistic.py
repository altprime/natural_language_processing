import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


message_data = pd.read_csv("./datasets/spam.csv", encoding="latin")
message_data.head()

message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
message_data = message_data.rename(columns={'v1': 'label', 'v2': 'message'})

message_data_copy = message_data['message'].copy()

# preprocessing
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

message_data_copy = message_data_copy.apply(text_preprocess)
message_data_copy

vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)
message_mat
xtrain, xtest, ytrain, ytest = train_test_split(message_mat, message_data['label'], test_size=0.3, random_state=20)

model = LogisticRegression(solver='liblinear', penalty='l1')
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

'''
                    ./results/accuracy_score.png
'''
accuracy_score(ytest, pred)

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

message_data_copy = message_data_copy.apply(stemmer)
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)

xtrain, xtest, ytrain, ytest = train_test_split(message_mat, message_data['label'], test_size=0.3, random_state=20)

model = LogisticRegression(solver='liblinear', penalty='l1')
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
'''
                    ./results/improved_accuracy_score.png
'''
accuracy_score(ytest, pred)


