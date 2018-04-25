# NATURAL LANGUAGE PROCESSING

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('/Users/karthikmudlapur/Desktop/Machine Learning A-Z /Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Natural_Language_Processing')

# import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# since we are importing a tsv file, we have to give the delimiter and quoting 3 is to remove double quotes

# cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000): 
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    # consider first record. We have to keep only the letters and remove all other characters
    review = review.lower()
    # to convert all the letters into lower case
    
    review = review.split()
    # to convert string to list(all the words into diff list) 
    
    ps = PorterStemmer()
    # stemming - Keeping only the base word and remove tense and other details(reduce sparcity)
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # creating a for loop to read all the lists(words) and remove the filler words which are in stopwords 
    
    review = ' '.join(review)
    # to attach all the list back into a string for a each record
    
    corpus.append(review)

# Creating the bag of words model (using tocanisation)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()  
# to create the sparse matrix  
y = dataset.iloc[:,1].values
# to create the dependedent variable

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/200