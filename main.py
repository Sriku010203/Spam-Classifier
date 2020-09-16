## SPAM CLASSIFIER

## Importing the data
import pandas as pd

messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['labels','Messages'])

## Data Cleaning

import nltk  ## Used for text preprocessing
import re    ## re->regular expressions
from nltk.corpus import stopwords   ## Stopwords-> which doesnt play a major role in the sentence
from nltk.stem import PorterStemmer,WordNetLemmatizer  ## importing steming and lematizer 

ps=PorterStemmer()  ## initializing stemmer obj
WordNet=WordNetLemmatizer() ## initilizing the Lemmatizer
ps_list=[]
for i in range(len(messages)):
    message=re.sub('[^a-zA-Z]',' ',messages['Messages'][i]) ##Considering only the alphabets and removing all others
    message=message.lower()
    message=message.split()
    message=[ps.stem(word) for word in message if word not in set(stopwords.words('english'))]
    message= ' '.join(message)
    ps_list.append(message)
    
from sklearn.feature_extraction.text import CountVectorizer ## Bag of words
CV=CountVectorizer(max_features=5000)   ## initializing 
X=CV.fit_transform(ps_list).toarray() ## converting all the text to values ->independent variables

y=pd.get_dummies(messages['labels'],drop_first=True)
y

from sklearn.model_selection import train_test_split  ## Splitting the train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=0.8)

## NAive bayes classifier ->  does classification bas on probability

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)

y_pred=model.predict(X_test)
from sklearn.metrics import classification_report

report=classification_report(y_test,y_pred)















    

    
    
    

